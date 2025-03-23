import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .types import MODEL_TYPE


@torch.no_grad()
def evaluate_ppl(model: MODEL_TYPE, input_tensor: torch.Tensor, device: str) -> float:
    def batch_nll(model: MODEL_TYPE, batch: torch.Tensor) -> torch.Tensor:
        logits = model(batch).logits[:, :-1, :].permute(0, 2, 1)
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits, labels, reduction="none")
        return loss.float().mean(dim=1)

    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)

    model.to(device).eval()
    nlls = [
        batch_nll(model, input_tensor[i].to(device))
        for i in tqdm(range(input_tensor.shape[0]), desc="Evaluating PPL")
    ]
    ppl = torch.exp(torch.cat(nlls).mean())
    return ppl.item()


@torch.no_grad()
def evaluate_zero_shot(
    model: MODEL_TYPE,
    tokenizer: PreTrainedTokenizerBase,
    task_names: list[str],
    batch_size: int,
    device: str,
) -> dict[str, float]:
    from lm_eval.models.huggingface import HFLM

    import lm_eval
    from lm_eval.tasks import TaskManager

    model.to(device).eval()
    hflm = HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device
    )
    tasks = TaskManager().match_tasks(task_names)
    results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size=batch_size)[
        "results"
    ]
    metric_vals = {
        task: round(result.get("acc_norm,none", result["acc,none"]), 4)
        for task, result in results.items()
    }
    metric_vals["acc_avg"] = round(
        sum(metric_vals.values()) / len(metric_vals.values()), 4
    )
    return metric_vals
