from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
import transformers

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .types import MODEL_TYPE


class ModelType(str, Enum):
    LLAMA_2_7B = "meta-llama/Llama-2-7b-hf"
    LLAMA_2_13B = "meta-llama/Llama-2-13b-hf"
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B"
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
    MISTRAL_7B = "mistralai/Mistral-7B-v0.3"


def load_model(
    model_type: ModelType,
    seqlen: int,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    return_tokenizer: bool = False,
    hf_token: str | None = None,
) -> tuple[MODEL_TYPE, PreTrainedTokenizerBase | None]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_type.value, torch_dtype=dtype, device_map=device_map, token=hf_token
    )
    model.seqlen = seqlen
    if return_tokenizer:
        tokenizer = load_tokenizer(model_type)
        return model, tokenizer
    return model, None


def load_tokenizer(
    model_type: ModelType,
) -> PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(model_type.value)
