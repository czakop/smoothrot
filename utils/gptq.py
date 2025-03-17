from __future__ import annotations

import logging
import math
import typing
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from .quant import Quantizer
    from .types import DECODER_LAYER_TYPE, MODEL_TYPE

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer: torch.nn.Linear, quantizer: Quantizer):
        self.layer = layer
        self.quantizer = quantizer
        self.dev = self.layer.weight.device
        self.columns = layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.t().matmul(inp)

    def quantize(
        self,
        blocksize=128,
        percdamp=0.01,
        actorder=False,
    ):
        W = self.layer.weight.data.clone().float()

        groupsize = self.quantizer.group_size
        if groupsize <= 0:
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        H_inv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            H_inv1 = H_inv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = H_inv1[i, i]

                if groupsize > 0:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i) : (i1 + i + groupsize)]
                        )

                q = self.quantizer(
                    w.unsqueeze(1), enable_reshape=False, return_dtype=torch.float32
                ).flatten()
                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(H_inv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(H_inv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


class Catcher(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, input_shape: tuple[int, ...]):
        super().__init__()
        self.module = module
        self.batch_idx = 0
        self.attention_mask = None
        self.position_embeddings = None
        dtype = next(iter(module.parameters())).dtype
        self.register_buffer("inputs", torch.zeros(*input_shape, dtype=dtype))

    def forward(self, inp: torch.Tensor, **kwargs: Any):
        self.inputs[self.batch_idx] = inp
        self.batch_idx += 1
        if not self.attention_mask:
            self.attention_mask = kwargs["attention_mask"]
        if not self.position_embeddings:
            self.position_embeddings = kwargs["position_embeddings"]
        raise ValueError("Stop forward pass")


def capture_parameters(
    model: MODEL_TYPE, input_ids: torch.Tensor, dev: str, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    model.model.embed_tokens.to(dev)
    model.model.rotary_emb.to(dev)
    nbatches = input_ids.numel() // (batch_size * model.seqlen)
    first_layer_catcher = Catcher(
        model.model.layers[0],
        (nbatches, batch_size, model.seqlen, model.config.hidden_size),
    )
    model.model.layers[0] = first_layer_catcher.to(dev)

    for batch in input_ids:
        try:
            model(batch.to(dev))
        except ValueError:
            pass

    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cpu()
    model.model.layers[0] = first_layer_catcher.module.cpu()

    torch.cuda.empty_cache()

    return (
        first_layer_catcher.inputs,
        first_layer_catcher.attention_mask,
        first_layer_catcher.position_embeddings,
    )


def get_nested_attr(obj, attr):
    for name in attr.split("."):
        obj = getattr(obj, name)
    return obj


@torch.no_grad()
def gptq_fwrd(
    model: MODEL_TYPE,
    inputs_ids: torch.Tensor,
    w_quantizer: Quantizer,
    percdamp: float = 0.01,
    act_order: bool = False,
    batch_size: int = 1,
    dev: str = "cuda",
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers: typing.Sequence[DECODER_LAYER_TYPE] = model.model.layers

    inps, attention_mask, position_embeddings = capture_parameters(
        model, inputs_ids, dev, batch_size
    )

    outs = torch.zeros_like(inps)

    sequential = [
        [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
        ],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    for i in tqdm(range(len(layers)), desc="GPTQ Quant"):
        layer = layers[i].to(dev)
        for names in sequential:
            subset = {n: get_nested_attr(layer, n) for n in names}

            gptq = {}
            handles = []

            def add_batch_hook(name):
                def hook(m, inp, out):
                    gptq[name].add_batch(inp[0].data)

                return hook

            for name in subset:
                gptq[name] = GPTQ(subset[name], w_quantizer)
                handles.append(subset[name].register_forward_hook(add_batch_hook(name)))

            for j, batch in enumerate(inps):
                outs[j] = layer(
                    batch,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                gptq[name].quantize(percdamp=percdamp, actorder=act_order)
                gptq[name].free()

        for j, batch in enumerate(inps):
            outs[j] = layer(
                batch,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
