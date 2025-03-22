import argparse
import functools
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

from .dataset import get_wikitext2
from .gptq import gptq_fwrd
from .hadamard import hadamard_transform
from .model import load_tokenizer
from .monkeypatch import add_wrapper_after_function_call_in_method
from .types import CONFIG_TYPE, MODEL_TYPE


@dataclass(frozen=True)
class QuantConfig:
    bits: int = 16
    dynamic: bool = True
    asym: bool = False
    per_tensor: bool = False
    group_size: int = -1
    clip_ratio: float = 1.0
    clip_opt: bool = False
    max_shrink: float = 0.8
    grid: int = 100

    def __post_init__(self):
        assert 0 < self.bits <= 16, "Number of bits should be in (0, 16]"
        assert 0 < self.clip_ratio <= 1, "Clipping ratio should be in (0, 1]"
        assert (
            not self.clip_opt or self.clip_ratio == 1.0
        ), "Clip ratio should be 1.0 in optimized mode"
        assert 0 <= self.max_shrink <= 1, "Max shrink should be in [0, 1]"
        assert self.grid > 0, "Grid size should be positive"


class Quantizer(torch.nn.Module):
    def __init__(self, config: QuantConfig = QuantConfig()):
        super(Quantizer, self).__init__()
        self._configure(config)
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))

    def _configure(self, config: QuantConfig = QuantConfig()):
        self.bits = config.bits
        self.dynamic = config.dynamic
        self.asym = config.asym
        self.per_tensor = config.per_tensor
        self.group_size = config.group_size
        self.clip_ratio = config.clip_ratio
        self.clip_opt = config.clip_opt
        self.max_shrink = config.max_shrink
        self.grid = config.grid

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_tensor:
            return x.reshape(-1, x.shape[-2] * x.shape[-1])
        elif self.group_size > 0:
            return x.reshape(-1, self.group_size)
        else:
            return x.reshape(-1, x.shape[-1])

    def _get_min_max(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.asym:
            maxq = torch.tensor(2**self.bits - 1, device=device)
            minq = torch.zeros(1, device=device)
        else:
            maxq = torch.tensor(2 ** (self.bits - 1) - 1, device=device)
            minq = -maxq - 1
        return minq, maxq

    def _find_params(
        self, xmin: torch.Tensor, xmax: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, maxq = self._get_min_max(device)

        if self.asym:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp], xmax[tmp] = -1, 1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            scale = scale.unsqueeze(1)
            zero = zero.unsqueeze(1)
        else:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / maxq).unsqueeze(1)
            scale[tmp] = 1
            zero = torch.zeros_like(scale, device=device)
        return scale, zero

    def _quant_dequant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        zero: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scale is None or zero is None:
            scale = self.scale
            zero = self.zero
        minq, maxq = self._get_min_max(x.device)
        x_int = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
        return scale * (x_int - zero)

    def calibrate(self, x: torch.Tensor) -> None:
        if self.bits == 16:
            return

        reshaped_x = self._reshape_input(x)

        tmp = torch.zeros(reshaped_x.shape[0], device=x.device)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio

        self.scale, self.zero = self._find_params(xmin, xmax, x.device)

        if self.clip_opt:
            best = torch.full([reshaped_x.shape[0]], float("inf"), device=x.device)
            for i in range(int(self.max_shrink * self.grid)):
                p = 1 - i / self.grid
                xmin_i = p * xmin
                xmax_i = p * xmax

                scale_i, zero_i = self._find_params(xmin_i, xmax_i, x.device)
                q_i = self._quant_dequant(reshaped_x, scale_i, zero_i)
                err_i = torch.norm(reshaped_x - q_i, p=2.4, dim=1)

                tmp = err_i < best
                if torch.any(tmp):
                    best[tmp] = err_i[tmp]
                    self.scale[tmp] = scale_i[tmp]
                    self.zero[tmp] = zero_i[tmp]

    def forward(
        self,
        x: torch.Tensor,
        enable_reshape: bool = True,
        return_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        if self.bits == 16:
            return x

        reshaped_x = self._reshape_input(x) if enable_reshape else x

        if (
            self.dynamic
            or not self.scale.count_nonzero()
            or not self.zero.count_nonzero()
        ):
            self.calibrate(reshaped_x)

        return self._quant_dequant(reshaped_x).reshape(x.shape).to(return_dtype)


class LinearWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(LinearWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear), "Only nn.Linear is supported"
        self.module = module
        self.in_quantizer = None
        self.out_quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_quantizer:
            x = self.in_quantizer(x)
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
        dynamic=False,
        asym=args.w_asym,
        per_tensor=args.w_per_tensor,
        group_size=args.w_group_size,
        clip_ratio=1.0,
        clip_opt=True,
    )
    a_quant_config = QuantConfig(
        bits=args.a_bits,
        asym=args.a_asym,
        per_tensor=args.a_per_tensor,
        group_size=args.a_group_size,
        clip_ratio=args.a_clip_ratio,
    )
    v_quant_config = QuantConfig(
        bits=args.v_bits,
        asym=args.v_asym,
        per_tensor=False,
        group_size=(
            config.hidden_size // config.num_attention_heads if args.v_per_head else -1
        ),
        clip_ratio=args.v_clip_ratio,
    )
    k_quant_config = QuantConfig(
        bits=args.k_bits,
        asym=args.k_asym,
        per_tensor=False,
        group_size=(
            config.hidden_size // config.num_attention_heads if args.k_per_head else -1
        ),
        clip_ratio=args.k_clip_ratio,
    )
    return w_quant_config, a_quant_config, v_quant_config, k_quant_config


def add_linear_wrappers(module: torch.nn.Module) -> None:
    if isinstance(module, LinearWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if isinstance(tmp, torch.nn.Linear):
            setattr(module, attr, LinearWrapper(tmp))
        elif isinstance(tmp, (torch.nn.Sequential, torch.nn.ModuleList)):
            tmp._modules = {
                name: (LinearWrapper(m) if isinstance(m, torch.nn.Linear) else m)
                for name, m in tmp.named_children()
            }
    for child in module.children():
        add_linear_wrappers(child)


def apply_to_linear_wrappers(
    module: torch.nn.Module, fn: Callable[[LinearWrapper], None]
) -> None:
    if isinstance(module, LinearWrapper):
        fn(module)
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if isinstance(tmp, LinearWrapper):
            fn(tmp)
        elif isinstance(tmp, (torch.nn.Sequential, torch.nn.ModuleList)):
            for m in tmp.children():
                apply_to_linear_wrappers(m, fn)
    for child in module.children():
        apply_to_linear_wrappers(child, fn)


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
    if args.gptq:
        calib_data = get_wikitext2(
            load_tokenizer(args.model),
            args.seqlen,
            args.gptq_calib_samples,
            args.batch_size,
            False,
            args.seed,
            "cpu",
        )
        gptq_fwrd(
            model,
            calib_data,
            Quantizer(w_quant_config),
            dev=device,
            batch_size=args.batch_size,
        )

    def rtn_fwrd(lin: LinearWrapper):
        lin.module.weight.data = Quantizer(w_quant_config)(lin.module.weight.data)

    layers: torch.nn.ModuleList = model.model.layers
    for i in tqdm(range(len(layers)), desc="Quantizing decoder layers"):
        layer = layers[i].to(device)
        if not args.gptq:
            apply_to_linear_wrappers(layer, rtn_fwrd)
        apply_to_linear_wrappers(
            layer, lambda lin: setattr(lin, "in_quantizer", Quantizer(a_quant_config))
        )
        add_post_rope_wrapper(layer.self_attn, k_quant_config, args.k_rotate)
        layer.self_attn.v_proj.out_quantizer = Quantizer(v_quant_config)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
