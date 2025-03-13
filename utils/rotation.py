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
def rotate_model(model: MODEL_TYPE, device: str = "cpu") -> None:
    Q = hadamard_matrix(model.config.hidden_size, random=True, device=device)
    num_heads: int = model.config.num_attention_heads
    model_dim: int = model.config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q, device)
    rotate_head(model, Q, device)
    torch.cuda.empty_cache()

    layers: typing.Sequence[DECODER_LAYER_TYPE] = model.model.layers
    for idx in tqdm(range(len(layers)), desc="Rotating"):
        rotate_attention_inputs(layers[idx], Q, device)
        rotate_attention_output(layers[idx], Q, device)
        rotate_mlp_input(layers[idx], Q, device)
        rotate_mlp_output(layers[idx], Q, device)
        rotate_ov_proj(layers[idx], head_dim)


def rotate_linear(
    linear: torch.nn.Linear, Q: torch.Tensor, output: bool = False, device: str = "cpu"
) -> None:
    dtype = linear.weight.dtype
    W_ = linear.weight.data.to(device=device, dtype=torch.float64)
    if output:
        linear.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    else:
        linear.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
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
    apply_exact_had_to_linear(
        down_proj, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output


def rotate_ov_proj(layer: DECODER_LAYER_TYPE, head_dim: int):
    v_proj: torch.nn.Linear = layer.self_attn.v_proj
    o_proj: torch.nn.Linear = layer.self_attn.o_proj
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
