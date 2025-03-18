import typing

import torch
from tqdm import tqdm

from .hadamard import apply_exact_had_to_linear, hadamard_matrix, hadamard_transform
from .types import DECODER_LAYER_TYPE, LAYERNORM_TYPE, MODEL_TYPE


def online_had_hook_factory(had_dim: int = -1) -> typing.Callable:
    def online_had_hook(
        module: torch.nn.Module, inputs: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor]:
        x = inputs[0]
        if had_dim > 0:
            return (
                (
                    hadamard_transform(
                        x.reshape(-1, x.shape[-1] // had_dim, had_dim).transpose(1, 2),
                        device=x.device,
                    )
                    .transpose(1, 2)
                    .reshape(x.shape)
                ),
            )
        else:
            return (hadamard_transform(x, device=x.device),)

    return online_had_hook


def fuse_ln_linear(
    layernorm: LAYERNORM_TYPE, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def reset_layernorm(layernorm: LAYERNORM_TYPE) -> None:
    # return layernorm.__class__(layernorm.weight.numel(), eps=layernorm.variance_epsilon)
    layernorm.weight.data = torch.ones_like(layernorm.weight.data)


@torch.no_grad()
def fuse_layernorms(model: MODEL_TYPE) -> None:
    # Untie word embeddings
    if model.config.tie_word_embeddings:
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(
            model.model.embed_tokens.weight.clone()
        )

    # Embedding fusion
    embedding: torch.nn.Embedding = model.model.embed_tokens
    W_ = embedding.weight.data.double()
    embedding.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(
        embedding.weight.data.dtype
    )

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in model.model.layers:
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )

        reset_layernorm(layer.post_attention_layernorm)
        reset_layernorm(layer.input_layernorm)

    # Fuse the linear operations in the final layer norm and the head.
    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )
    reset_layernorm(model.model.norm)


@torch.no_grad()
def rotate_model(
    model: MODEL_TYPE,
    spinquant: bool = False,
    spinquant_r_path: str | None = None,
    device: str = "cpu",
) -> None:
    num_heads: int = model.config.num_attention_heads
    model_dim: int = model.config.hidden_size
    head_dim = model_dim // num_heads
    if spinquant:
        assert spinquant_r_path is not None, "SpinQuant rotation path is not provided."
        R = torch.load(spinquant_r_path)
        Q = R["R1"].to(device=device, dtype=torch.float64)
        assert Q.shape == (model_dim, model_dim), "Invalid R1 shape."
    else:
        Q = hadamard_matrix(model_dim, random=True, device=device)

    rotate_embeddings(model, Q, device)
    rotate_head(model, Q, device)
    torch.cuda.empty_cache()

    layers: typing.Sequence[DECODER_LAYER_TYPE] = model.model.layers
    for idx in tqdm(range(len(layers)), desc="Rotating"):
        rotate_attention_inputs(layers[idx], Q, device)
        rotate_attention_output(layers[idx], Q, device)
        rotate_mlp_input(layers[idx], Q, device)
        rotate_mlp_output(layers[idx], Q, device)
        if spinquant:
            R2 = R[f"model.layers.{idx}.self_attn.R2"].to(
                device=device, dtype=torch.float64
            )
            assert R2.shape == (head_dim, head_dim), "Invalid R2 shape."
        else:
            R2 = None
        rotate_ov_proj(layers[idx], R2, head_dim, device)


def rotate_linear(
    linear: torch.nn.Linear,
    Q: torch.Tensor | None = None,
    output: bool = False,
    rot_dim: int = -1,
    device: str = "cpu",
) -> None:
    dtype = linear.weight.dtype
    if Q is None:
        apply_exact_had_to_linear(
            linear,
            had_dim=rot_dim,
            output=output,
            device=device,
        )
        return

    W_ = linear.weight.data.to(device=device, dtype=torch.float64)

    if rot_dim == -1:
        if output:
            W_ = torch.matmul(Q.T, W_)
        else:
            W_ = torch.matmul(W_, Q)
    else:
        if output:
            W_ = W_.t()
            W_ = (
                torch.matmul(W_.reshape(-1, W_.shape[-1] // rot_dim, rot_dim), Q)
                .reshape(W_.shape)
                .t()
            )
        else:
            W_ = torch.matmul(
                W_.reshape(-1, W_.shape[-1] // rot_dim, rot_dim), Q
            ).reshape(W_.shape)

    linear.weight.data = W_.to(device="cpu", dtype=dtype)

    if hasattr(linear, "bias") and linear.bias is not None:
        b = linear.bias.data.to(device=device, dtype=torch.float64)
        if output:
            linear.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
        else:
            linear.bias.data = torch.matmul(b, Q).to(device="cpu", dtype=dtype)


def rotate_embeddings(model: MODEL_TYPE, Q: torch.Tensor, device: str = "cpu") -> None:
    rotate_linear(model.model.embed_tokens, Q, device=device)


def rotate_head(model: MODEL_TYPE, Q: torch.Tensor, device: str = "cpu") -> None:
    rotate_linear(model.lm_head, Q, device=device)


def rotate_attention_inputs(
    layer: DECODER_LAYER_TYPE, Q: torch.Tensor, device: str = "cpu"
) -> None:
    for linear in [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ]:
        rotate_linear(linear, Q, device=device)


def rotate_attention_output(
    layer: DECODER_LAYER_TYPE, Q: torch.Tensor, device: str = "cpu"
) -> None:
    rotate_linear(layer.self_attn.o_proj, Q, output=True, device=device)


def rotate_mlp_input(
    layer: DECODER_LAYER_TYPE, Q: torch.Tensor, device: str = "cpu"
) -> None:
    for linear in [layer.mlp.up_proj, layer.mlp.gate_proj]:
        rotate_linear(linear, Q, device=device)


def rotate_mlp_output(
    layer: DECODER_LAYER_TYPE, Q: torch.Tensor, device: str = "cpu"
) -> None:
    down_proj: torch.nn.Linear = layer.mlp.down_proj
    rotate_linear(down_proj, Q, output=True, device=device)
    rotate_linear(down_proj, None, output=False, rot_dim=-1, device=device)


def rotate_ov_proj(
    layer: DECODER_LAYER_TYPE,
    R2: torch.Tensor | None,
    head_dim: int,
    device: str = "cpu",
) -> None:
    v_proj: torch.nn.Linear = layer.self_attn.v_proj
    o_proj: torch.nn.Linear = layer.self_attn.o_proj
    rotate_linear(v_proj, R2, output=True, rot_dim=head_dim, device=device)
    rotate_linear(
        o_proj,
        R2,
        output=False,
        rot_dim=head_dim if R2 is not None else -1,
        device=device,
    )
