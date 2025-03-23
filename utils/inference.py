from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch

if TYPE_CHECKING:
    from .types import MODEL_TYPE


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
    model: MODEL_TYPE, input_ids: torch.Tensor, dev: str
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    model.model.embed_tokens.to(dev)
    model.model.rotary_emb.to(dev)
    nbatches = input_ids.numel() // (input_ids.shape[-2] * model.seqlen)
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


class ActivationRegistry:
    _input_modules: list[str] = list()
    _output_modules: list[str] = list()
    _input_registry: dict[str, list[torch.Tensor]] = dict()
    _output_registry: dict[str, list[torch.Tensor]] = dict()

    @classmethod
    def set_input_modules(cls, input_modules: list[str]):
        cls._input_modules = input_modules

    @classmethod
    def set_output_modules(cls, output_modules: list[str]):
        cls._output_modules = output_modules

    @classmethod
    def empty(cls) -> None:
        cls._input_registry = {module_name: [] for module_name in cls._input_modules}
        cls._output_registry = {module_name: [] for module_name in cls._output_modules}

    @classmethod
    def register_input(cls, module_name: str, tensor: torch.Tensor):
        cls._input_registry[module_name].append(tensor)

    @classmethod
    def register_output(cls, module_name: str, tensor: torch.Tensor):
        cls._output_registry[module_name].append(tensor)

    @classmethod
    def save(cls, folder: Path | str, prefix: str = "", empty: bool = True):
        if isinstance(folder, str):
            folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        for module_name in cls._input_registry:
            if cls._input_registry[module_name]:
                torch.save(
                    torch.cat(cls._input_registry[module_name], dim=0),
                    folder / f"{prefix}{module_name}_in.pt",
                )
        for module_name in cls._output_registry:
            if cls._output_registry[module_name]:
                torch.save(
                    torch.cat(cls._output_registry[module_name], dim=0),
                    folder / f"{prefix}{module_name}_out.pt",
                )
        if empty:
            cls.empty()


def hook_factory(module_name: str):
    def hook(m: torch.nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor):
        ActivationRegistry.register_input(module_name, input[0].detach().cpu())

    return hook


@contextmanager
def capture_down_proj_input(
    model: MODEL_TYPE,
    layers: list[int] | None,
    save_folder: Path | str,
    prefix: str = "",
) -> Iterator[None]:
    if layers is None:
        layers = list(range(len(model.model.layers)))

    ActivationRegistry.set_input_modules([f"down_{layer_idx}" for layer_idx in layers])
    ActivationRegistry.empty()

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx in layers:
        module = model.model.layers[layer_idx].mlp.down_proj
        hook_handles.append(
            module.register_forward_hook(hook_factory(f"down_{layer_idx}"))
        )
    try:
        yield
    finally:
        for h in hook_handles:
            h.remove()
        ActivationRegistry.save(save_folder, prefix, empty=True)
