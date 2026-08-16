import torch
from torch import nn


class ExpertMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, horizon: int, dropout: float, future_feature_dim: int = 0):
        super().__init__()
        self.future_feature_dim = future_feature_dim
        output_dim = 1 if future_feature_dim else horizon
        self.net = nn.Sequential(nn.Linear(latent_dim + future_feature_dim, latent_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x, future_features=None):
        if self.future_feature_dim:
            if future_features is None:
                future_features = x.new_zeros(x.size(0), 1, self.future_feature_dim)
            x = x.unsqueeze(1).expand(-1, future_features.size(1), -1)
            x = torch.cat([x, future_features], dim=-1)
        return self.net(x).squeeze(-1)
