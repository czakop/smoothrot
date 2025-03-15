from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass(frozen=True)
class QuantConfig:
    bits: int = 8
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
        out_quant: bool = False,
    ):
        super(QuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear), "Only nn.Linear is supported"
        module.weight.data = Quantizer(w_quant_config)(module.weight.data)
        self.module = module
        self.quantizer = Quantizer(a_quant_config)
        self.out_quant = out_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer(x)
        y = self.module(x)
        return self.quantizer(y) if self.out_quant else y


def add_quant(
    model: torch.nn.Module, w_quant_config: QuantConfig, a_quant_config: QuantConfig
):
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
        add_quant(child, w_quant_config, a_quant_config)


@torch.no_grad()
def quantize_model(
    model: torch.nn.Module,
    w_quant_config: QuantConfig,
    a_quant_config: QuantConfig,
    device: str = "cuda",
):
    model.eval()
    layers: torch.nn.ModuleList = model.model.layers
    for i in tqdm(range(len(layers)), desc="Quantizing decoder layers"):
        layer = layers[i].to(device)
        add_quant(layer, w_quant_config, a_quant_config)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
