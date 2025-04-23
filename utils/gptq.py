from __future__ import annotations

import logging
import math
import typing
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from .inference import capture_parameters

if TYPE_CHECKING:
    from .quant import Quantizer
    from .types import DECODER_LAYER_TYPE, MODEL_TYPE

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layers: list[torch.nn.Linear], quantizer: Quantizer):
        self.layers = layers
        self.quantizer = quantizer
        self.dev = self.layers[0].weight.device
        self.columns = layers[0].weight.data.shape[1]
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
        W = torch.cat([layer.weight.data for layer in self.layers], dim=0).float()

        groupsize = self.quantizer.group_size
        if groupsize <= 0:
            self.quantizer.calibrate(W)

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
                        self.quantizer.calibrate(W[:, (i1 + i) : (i1 + i + groupsize)])

                q = self.quantizer(
                    w.unsqueeze(1),
                    enable_reshape=False,
                    return_dtype=torch.float32,
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

        sizes = [layer.weight.data.shape[0] for layer in self.layers]
        split_weights = torch.split(Q, sizes, dim=0)
        for layer, split_weight in zip(self.layers, split_weights):
            layer.weight.data.copy_(split_weight.to(layer.weight.data.dtype))
            if torch.any(torch.isnan(layer.weight.data)):
                logging.warning("NaN in weights")
                raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


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
    dev: str = "cuda",
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers: typing.Sequence[DECODER_LAYER_TYPE] = model.model.layers

    inps, attention_mask, position_embeddings = capture_parameters(
        model, inputs_ids, dev
    )

    outs = torch.zeros_like(inps)

    modules = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]
    for i in tqdm(range(len(layers)), desc="GPTQ Quant"):
        layer = layers[i].to(dev)

        gptq = {}
        handles = []

        def add_batch_hook(id):
            def hook(m, inp, out):
                gptq[id].add_batch(inp[0].data)

            return hook

        for m_id, names in enumerate(modules):
            subset = [get_nested_attr(layer, n) for n in names]

            gptq[m_id] = GPTQ(subset, w_quantizer)
            handles.append(subset[0].register_forward_hook(add_batch_hook(m_id)))

        for j, batch in enumerate(inps):
            outs[j] = layer(
                batch,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

        for h in handles:
            h.remove()

        for m_id in gptq:
            gptq[m_id].quantize(percdamp=percdamp, actorder=act_order)
            gptq[m_id].free()

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
