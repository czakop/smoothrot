from __future__ import annotations

import random
from enum import Enum
from typing import TYPE_CHECKING

import datasets
import torch
import transformers

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class DatasetType(str, Enum):
    WIKITEXT2 = "wikitext2"
    C4_NEW = "c4_new"
    PTB_NEW = "ptb_new"
    RANDOM = "random"


def get_wikitext2(
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    num_samples: int = -1,
    batch_size: int = 1,
    eval_mode: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    data = datasets.load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test" if eval_mode else "train"
    )
    input_ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    if num_samples > 0:
        input_ids = _sample_input_ids(input_ids, seqlen, num_samples)
    return _reshape_and_truncate(input_ids, seqlen, batch_size, device)


def get_c4_new(
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    num_samples: int = -1,
    batch_size: int = 1,
    eval_mode: bool = True,
    device: str = "cpu",
) -> torch.Tensor:

    if eval_mode:
        data = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        input_ids = tokenizer(
            " ".join(data[:1100]["text"]), return_tensors="pt"
        ).input_ids[:, : (256 * seqlen)]
        if num_samples > 0:
            input_ids = _sample_input_ids(input_ids, seqlen, num_samples)
    else:
        assert num_samples > 0, "num_samples must be > 0 for training"
        data = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )

        samples = []
        for _ in range(num_samples):
            while True:
                i = random.randint(0, len(data) - 1)
                input_ids = tokenizer(data[i]["text"], return_tensors="pt").input_ids
                if input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, input_ids.shape[1] - seqlen - 1)
            samples.append(input_ids[:, i : i + seqlen])
        input_ids = torch.cat(samples, dim=0)
    return _reshape_and_truncate(input_ids, seqlen, batch_size, device)


def get_ptb_new(
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    num_samples: int = -1,
    batch_size: int = 1,
    eval_mode: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    data = datasets.load_dataset(
        "ptb_text_only",
        "penn_treebank",
        split="test" if eval_mode else "train",
        trust_remote_code=True,
    )
    input_ids = tokenizer(" ".join(data["sentence"]), return_tensors="pt").input_ids
    if num_samples > 0:
        input_ids = _sample_input_ids(input_ids, seqlen, num_samples)
    return _reshape_and_truncate(input_ids, seqlen, batch_size, device)


def get_random(
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    num_samples: int = -1,
    batch_size: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    assert num_samples > 0, "num_samples must be > 0 for training"
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, num_samples * seqlen))
    return _reshape_and_truncate(input_ids, seqlen, batch_size, device)


def load_dataset(
    dataset_type: DatasetType,
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    num_samples: int = -1,
    batch_size: int = 1,
    eval_mode: bool = True,
    device: str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    transformers.set_seed(seed)
    match dataset_type:
        case DatasetType.WIKITEXT2:
            return get_wikitext2(
                tokenizer, seqlen, num_samples, batch_size, eval_mode, device
            )
        case DatasetType.C4_NEW:
            return get_c4_new(
                tokenizer, seqlen, num_samples, batch_size, eval_mode, device
            )
        case DatasetType.PTB_NEW:
            return get_ptb_new(
                tokenizer, seqlen, num_samples, batch_size, eval_mode, device
            )
        case DatasetType.RANDOM:
            assert not eval_mode, "RANDOM dataset is only for training"
            return get_random(tokenizer, seqlen, num_samples, batch_size, device)


def _reshape_and_truncate(
    input_ids: torch.Tensor, seqlen: int, batch_size: int, device: str
) -> torch.Tensor:
    num_batches = input_ids.numel() // seqlen // batch_size  # the tail is truncated
    return (
        input_ids[:, : num_batches * seqlen * batch_size]
        .view(-1, batch_size, seqlen)
        .to(device)
    )


def _sample_input_ids(
    input_ids: torch.Tensor, seqlen: int, num_samples: int
) -> torch.Tensor:
    samples = []
    for _ in range(num_samples):
        i = random.randint(0, input_ids.shape[1] - seqlen - 1)
        samples.append(input_ids[:, i : i + seqlen])
    return torch.cat(samples, dim=0)
