from torch import nn
from .moe_forecaster import MoEForecaster, ForecasterConfig


class DenseForecaster(MoEForecaster):
    def __init__(self, time_feature_dim: int, config: ForecasterConfig | None = None, n_series: int = 1):
        super().__init__(time_feature_dim, config, n_series); c = self.config
        self.head = nn.Sequential(nn.Linear(c.latent_dim, c.latent_dim), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(c.latent_dim, c.expert_hidden_dim), nn.GELU(), nn.Linear(c.expert_hidden_dim, c.horizon))

    def forward(self, window, features, series_index=None, return_diagnostics=False):
        h, mu, sigma = self.encode(window, features, series_index); forecast = self.head(h)
        from ..normalization import inverse_normalize
        return inverse_normalize(forecast, mu, sigma)
