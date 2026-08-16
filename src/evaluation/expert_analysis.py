from __future__ import annotations
import numpy as np
import torch


def window_statistics(window: torch.Tensor) -> dict[str, float]:
    x = window.detach().float().flatten(); mean = x.mean(); std = x.std(unbiased=False)
    recent = x[-1] - x[0] if x.numel() > 1 else torch.tensor(0.)
    return {"window_mean": mean.item(), "window_std": std.item(), "coefficient_of_variation": (std / mean.abs().clamp_min(1e-6)).item(), "zero_fraction": (x == 0).float().mean().item(), "recent_trend": recent.item(), "min": x.min().item(), "max": x.max().item(), "median": x.median().item()}


def routing_diagnostics(diagnostics: dict[str, torch.Tensor], warning_threshold=.8) -> dict[str, object]:
    probs = diagnostics["probabilities"]; selected = diagnostics["top_k_indices"]
    n = probs.size(-1); rates = torch.bincount(selected.flatten(), minlength=n).float() / selected.numel()
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1).mean()
    return {"selection_rate": rates, "mean_probability": probs.mean(0), "router_entropy": entropy, "warning": bool(rates.max() > warning_threshold)}


def diagnostic_rows(unique_ids, cutoffs, diagnostics):
    p, indices = diagnostics["probabilities"].detach().cpu(), diagnostics["top_k_indices"].detach().cpu()
    rows = []
    for i, (uid, cutoff) in enumerate(zip(unique_ids, cutoffs)):
        row = {"unique_id": uid, "cutoff_ds": cutoff}
        row.update({f"expert_{j + 1}_probability": float(p[i, j]) for j in range(p.size(1))})
        row.update({f"selected_expert_{j + 1}": int(indices[i, j]) + 1 for j in range(indices.size(1))})
        rows.append(row)
    return rows
