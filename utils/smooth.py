import functools

import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)


@torch.no_grad()
def get_act_scales(
    model: torch.nn.Module, input_ids: torch.Tensor, device: str = "cuda"
):
    model.to(device).eval()
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(input_ids.shape[0]), desc="Collecting activation scales"):
        model(input_ids[i].to(device))

    for h in hooks:
        h.remove()

    model.cpu()

    return act_scales


@torch.no_grad()
def smooth_act(
    prev_module: torch.nn.Module,
    fcs: torch.nn.Linear | list[torch.nn.Linear],
    act_scales: torch.Tensor,
    alpha: float = 0.5,
):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, torch.nn.Linear) and fc.in_features == act_scales.numel()

    if isinstance(prev_module, (LlamaRMSNorm, MistralRMSNorm)):
        assert prev_module.weight.numel() == act_scales.numel()
    elif isinstance(prev_module, torch.nn.Linear):
        assert prev_module.out_features == act_scales.numel()
    else:
        raise ValueError(f"Unsupported prev_module type '{type(prev_module)}'")

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight = torch.cat([fc.weight for fc in fcs], dim=0)
    weight_scales = weight.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    if isinstance(prev_module, (LlamaRMSNorm, MistralRMSNorm)):
        prev_module.weight.div_(scales)
    else:
        prev_module.weight.div_(scales.view(-1, 1))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


def smooth_model(
    model: torch.nn.Module, scales: dict[str, torch.Tensor], alpha: float = 0.5
):
    for name, module in model.named_modules():
        if not isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            continue

        # attention qkv projections
        attn_ln = module.input_layernorm
        qkv = [
            module.self_attn.q_proj,
            module.self_attn.k_proj,
            module.self_attn.v_proj,
        ]
        qkv_input_scales = scales[name + ".self_attn.q_proj"]
        smooth_act(attn_ln, qkv, qkv_input_scales, alpha)

        # mlp up and gate projections
        ffn_ln = module.post_attention_layernorm
        fcs = [module.mlp.gate_proj, module.mlp.up_proj]
        fcs_input_scales = scales[name + ".mlp.gate_proj"]
        smooth_act(ffn_ln, fcs, fcs_input_scales, alpha)

        # mlp down projection
        smooth_act(
            module.mlp.up_proj,
            module.mlp.down_proj,
            scales[name + ".mlp.down_proj"],
            alpha,
        )
