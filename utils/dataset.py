from __future__ import annotations

import random
from typing import TYPE_CHECKING

import datasets

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


def get_wikitext2(
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    num_samples: int = -1,
    batch_size: int = 1,
    eval_mode: bool = True,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    data = datasets.load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test" if eval_mode else "train"
    )
    input_ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    if num_samples > 0:
        random.seed(seed)
        i = random.randint(0, input_ids.shape[1] - num_samples * seqlen - 1)
        input_ids = input_ids[:, i : i + num_samples * seqlen]
    num_batches = input_ids.numel() // seqlen // batch_size  # the tail is truncated
    return (
        input_ids[:, : num_batches * seqlen * batch_size]
        .view(-1, batch_size, seqlen)
        .to(device)
    )
