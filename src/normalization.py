from __future__ import annotations

import torch


def normalize_window(window: torch.Tensor, epsilon: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = window.mean(dim=-1, keepdim=True)
    sigma = window.std(dim=-1, unbiased=False, keepdim=True)
    safe_sigma = sigma.clamp_min(epsilon)
    return (window - mu) / safe_sigma, mu, safe_sigma


def inverse_normalize(z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return z * sigma + mu
