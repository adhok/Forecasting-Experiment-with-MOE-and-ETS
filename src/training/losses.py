import torch
import torch.nn.functional as F


def forecasting_loss(prediction, target, kind="huber", scale=None):
    if scale is not None:
        error = (prediction - target) / scale.clamp_min(torch.finfo(prediction.dtype).eps)
        target = torch.zeros_like(error)
        prediction = error
    if kind == "mae": return F.l1_loss(prediction, target)
    if kind == "mse": return F.mse_loss(prediction, target)
    if kind == "huber": return F.huber_loss(prediction, target)
    raise ValueError(f"unknown loss: {kind}")


def routing_balance_loss(probabilities: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
    """Penalty encouraging both average probability and selections to be distributed."""
    n = probabilities.size(-1)
    importance = probabilities.mean(0)
    counts = torch.zeros(n, device=probabilities.device, dtype=probabilities.dtype)
    counts.scatter_add_(0, top_k_indices.reshape(-1), torch.ones(top_k_indices.numel(), device=probabilities.device, dtype=probabilities.dtype))
    load = counts / top_k_indices.numel()
    return n * torch.sum(importance * load)


def total_loss(prediction, target, diagnostics, kind="huber", balance_coefficient=0.01):
    forecast = forecasting_loss(prediction, target, kind, diagnostics.get("normalization_scale"))
    balance = routing_balance_loss(diagnostics["probabilities"], diagnostics["top_k_indices"])
    return {"forecast_loss": forecast, "routing_balance_loss": balance, "total_loss": forecast + balance_coefficient * balance}
