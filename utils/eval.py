import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, input_tensor: torch.Tensor, device: str
) -> float:
    def batch_nll(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(device)
        logits = model(batch).logits[:, :-1, :].permute(0, 2, 1)
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits, labels, reduction="none")
        return loss.float().mean(dim=1)

    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)

    model.eval()
    model = model.to(device)
    nlls = [
        batch_nll(model, input_tensor[i].to(device))
        for i in tqdm(range(input_tensor.shape[0]))
    ]
    ppl = torch.exp(torch.cat(nlls).mean())
    return ppl.item()
