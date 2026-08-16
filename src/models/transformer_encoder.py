from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        return self.encoder(self.projection(x))
