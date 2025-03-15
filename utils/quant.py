import argparse
import functools
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

from .hadamard import hadamard_transform
from .monkeypatch import add_wrapper_after_function_call_in_method
from .types import CONFIG_TYPE, MODEL_TYPE


@dataclass(frozen=True)
class QuantConfig:
    bits: int = 16
    asym: bool = True
    per_tensor: bool = False
    group_size: int = -1
    clipping_ratio: float = 1.0

    def __post_init__(self):
        assert 0 < self.bits <= 16, "Number of bits should be in (0, 16]"
        assert 0 < self.clipping_ratio <= 1, "Clipping ratio should be in (0, 1]"


class Quantizer(torch.nn.Module):
    def __init__(self, config: QuantConfig = QuantConfig()):
        super(Quantizer, self).__init__()
        self.configure(config)

    def configure(self, config: QuantConfig = QuantConfig()):
        self.bits = config.bits
        self.asym = config.asym
        self.per_tensor = config.per_tensor
        self.group_size = config.group_size
        self.clipping_ratio = config.clipping_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 16:
            return x

        init_shape = x.shape
        if self.per_tensor:
            reshaped_x = x.reshape(-1, x.shape[-2] * x.shape[-1])
        elif self.group_size > 0:
            reshaped_x = x.reshape(-1, self.group_size)
        else:
            reshaped_x = x.reshape(-1, x.shape[-1])

        tmp = torch.zeros(reshaped_x.shape[0], device=x.device)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clipping_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clipping_ratio

        if self.asym:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp], xmax[tmp] = -1, 1
            maxq = torch.tensor(2**self.bits - 1, device=x.device)
            minq = torch.zeros(1, device=x.device)
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            scale = scale.unsqueeze(1)
            zero = zero.unsqueeze(1)
        else:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            maxq = torch.tensor(2 ** (self.bits - 1) - 1, device=x.device)
            minq = -maxq - 1
            scale = (xmax / maxq).unsqueeze(1)
            scale[tmp] = 1
            zero = torch.zeros_like(scale, device=x.device)

        q = torch.clamp(torch.round(reshaped_x / scale) + zero, minq, maxq)
        return (scale * (q - zero)).reshape(init_shape).half()


class QuantWrapper(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        w_quant_config: QuantConfig,
        a_quant_config: QuantConfig,
    ):
        super(QuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear), "Only nn.Linear is supported"
        module.weight.data = Quantizer(w_quant_config)(module.weight.data)
        self.module = module
        self.quantizer = Quantizer(a_quant_config)
        self.out_quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer(x)
        y = self.module(x)
        return self.out_quantizer(y) if self.out_quantizer else y


class PostRoPEWrapper(torch.nn.Module):
    def __init__(self, rope_func: Callable, k_quant_config: QuantConfig, rotate: bool):
        super(PostRoPEWrapper, self).__init__()
        self.rope_func = rope_func
        self.k_quantizer = Quantizer(k_quant_config)
        self.rotate = rotate

    def forward(self, *args, **kwargs):
        q, k = self.rope_func(*args, **kwargs)
        if self.rotate:
            k = hadamard_transform(k, k.device)
            q = hadamard_transform(q, q.device)
        return q, self.k_quantizer(k)


def parse_quant_config(
    args: argparse.Namespace, config: CONFIG_TYPE
) -> tuple[QuantConfig, QuantConfig, QuantConfig, QuantConfig]:
    w_quant_config = QuantConfig(
        bits=args.w_bits,
        asym=args.w_asym,
        per_tensor=args.w_per_tensor,
        group_size=args.w_group_size,
        clipping_ratio=args.w_clip_ratio,
    )
    a_quant_config = QuantConfig(
        bits=args.a_bits,
        asym=args.a_asym,
        per_tensor=args.a_per_tensor,
        group_size=args.a_group_size,
        clipping_ratio=args.a_clip_ratio,
    )
    v_quant_config = QuantConfig(
        bits=args.v_bits,
        asym=args.v_asym,
        per_tensor=False,
        group_size=(
            config.hidden_size // config.num_attention_heads if args.v_per_head else -1
        ),
        clipping_ratio=args.v_clip_ratio,
    )
    k_quant_config = QuantConfig(
        bits=args.k_bits,
        asym=args.k_asym,
        per_tensor=False,
        group_size=(
            config.hidden_size // config.num_attention_heads if args.k_per_head else -1
        ),
        clipping_ratio=args.k_clip_ratio,
    )
    return w_quant_config, a_quant_config, v_quant_config, k_quant_config


def add_quant_wrapper(
    model: torch.nn.Module, w_quant_config: QuantConfig, a_quant_config: QuantConfig
) -> None:
    if isinstance(model, QuantWrapper):
        return
    for attr in dir(model):
        tmp = getattr(model, attr)
        if isinstance(tmp, torch.nn.Linear):
            setattr(model, attr, QuantWrapper(tmp, w_quant_config, a_quant_config))
        elif isinstance(tmp, (torch.nn.Sequential, torch.nn.ModuleList)):
            tmp._modules = {
                name: (
                    QuantWrapper(m, w_quant_config, a_quant_config)
                    if isinstance(m, torch.nn.Linear)
                    else m
                )
                for name, m in tmp.named_children()
            }
    for child in model.children():
        add_quant_wrapper(child, w_quant_config, a_quant_config)


def add_post_rope_wrapper(
    module: torch.nn.Module, k_quant_config: QuantConfig, rotate: bool
) -> None:
    rope_func_name = "apply_rotary_pos_emb"
    attr_name = "post_rope_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = add_wrapper_after_function_call_in_method(
        module,
        "forward",
        rope_func_name,
        functools.partial(
            PostRoPEWrapper, k_quant_config=k_quant_config, rotate=rotate
        ),
    )
    setattr(module, attr_name, wrapper)


@torch.no_grad()
def quantize_model(
    model: MODEL_TYPE,
    args: argparse.Namespace,
) -> None:
    model.eval()
    device = args.device
    w_quant_config, a_quant_config, v_quant_config, k_quant_config = parse_quant_config(
        args, model.config
    )
    layers: torch.nn.ModuleList = model.model.layers
    for i in tqdm(range(len(layers)), desc="Quantizing decoder layers"):
        layer = layers[i].to(device)
        add_quant_wrapper(layer, w_quant_config, a_quant_config)
        add_post_rope_wrapper(layer.self_attn, k_quant_config, args.k_rotate)
        layer.self_attn.v_proj.out_quantizer = Quantizer(v_quant_config)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
