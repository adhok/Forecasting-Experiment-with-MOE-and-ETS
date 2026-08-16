from torch import nn
import torch


class TopKRouter(nn.Module):
    def __init__(self, latent_dim: int, n_experts: int, top_k: int):
        super().__init__(); self.linear = nn.Linear(latent_dim, n_experts); self.top_k = top_k

    def forward(self, h):
        logits = self.linear(h); probabilities = logits.softmax(-1)
        values, indices = probabilities.topk(self.top_k, dim=-1)
        weights = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        return {"raw_logits": logits, "probabilities": probabilities, "top_k_indices": indices, "top_k_weights": weights}
