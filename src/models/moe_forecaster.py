from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from ..normalization import normalize_window, inverse_normalize
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer_encoder import TemporalEncoder
from .expert import ExpertMLP
from .router import TopKRouter


@dataclass
class ForecasterConfig:
    lookback: int = 52; horizon: int = 13; d_model: int = 64; n_heads: int = 4; n_transformer_layers: int = 2
    n_experts: int = 4; top_k: int = 2; latent_dim: int = 128; expert_hidden_dim: int = 64; series_embedding_dim: int = 16; dropout: float = .1; normalization_epsilon: float = 1e-6


class MoEForecaster(nn.Module):
    def __init__(self, time_feature_dim: int, config: ForecasterConfig | None = None, n_series: int = 1, future_feature_dim: int = 0):
        super().__init__(); self.config = config or ForecasterConfig()
        if not 1 <= self.config.top_k <= self.config.n_experts: raise ValueError("top_k must be in [1, n_experts]")
        c = self.config; self.positional = SinusoidalPositionalEncoding(c.d_model, c.lookback)
        self.temporal = TemporalEncoder(1 + time_feature_dim, c.d_model, c.n_heads, c.n_transformer_layers, c.dropout)
        self.series_embedding = nn.Embedding(n_series, c.series_embedding_dim)
        self.latent = nn.Linear(c.d_model + c.series_embedding_dim, c.latent_dim); self.router = TopKRouter(c.latent_dim, c.n_experts, c.top_k)
        self.future_feature_dim = future_feature_dim
        self.experts = nn.ModuleList([ExpertMLP(c.latent_dim, c.expert_hidden_dim, c.horizon, c.dropout, future_feature_dim) for _ in range(c.n_experts)])

    def encode(self, window, features, series_index=None):
        z, mu, sigma = normalize_window(window, self.config.normalization_epsilon)
        projected = self.temporal.projection(torch.cat([z.unsqueeze(-1), features], -1))
        encoded = self.temporal.encoder(self.positional(projected))
        # The final token attends to the complete history and preserves recent context
        # without collapsing all time steps into a mean.
        pooled = encoded[:, -1, :]
        if series_index is None:
            series_index = torch.zeros(window.size(0), dtype=torch.long, device=window.device)
        pooled = torch.cat([pooled, self.series_embedding(series_index)], dim=-1)
        return self.latent(pooled), mu, sigma

    def forward(self, window, features, series_index=None, return_diagnostics=True, future_features=None):
        h, mu, sigma = self.encode(window, features, series_index); diagnostics = self.router(h)
        diagnostics["normalization_scale"] = sigma
        # Sparse dispatch: each expert only processes samples routed to it.
        # This preserves per-sample top-k routing without evaluating all experts.
        forecast_normalized = torch.zeros(
            h.size(0), self.config.horizon, device=h.device, dtype=h.dtype
        )
        for expert_id, expert in enumerate(self.experts):
            batch_indices, top_k_slots = (
                diagnostics["top_k_indices"] == expert_id
            ).nonzero(as_tuple=True)
            if batch_indices.numel() == 0:
                continue
            expert_future_features = None if future_features is None else future_features[batch_indices]
            expert_output = expert(h[batch_indices], expert_future_features)
            routing_weights = diagnostics["top_k_weights"][batch_indices, top_k_slots]
            forecast_normalized.index_add_(
                0, batch_indices, expert_output * routing_weights.unsqueeze(-1)
            )
        forecast = inverse_normalize(forecast_normalized, mu, sigma)
        return (forecast, diagnostics) if return_diagnostics else forecast
