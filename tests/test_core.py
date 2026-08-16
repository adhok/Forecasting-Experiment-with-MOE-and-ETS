import pandas as pd
import torch
from src.data import PanelWindowDataset
from src.models import MoEForecaster, DenseForecaster, ForecasterConfig
from src.training.losses import total_loss
from src.evaluation.expert_analysis import routing_diagnostics


def test_windows_never_cross_series():
    f = pd.DataFrame({"unique_id": ["a"] * 6 + ["b"] * 6, "ds": list(pd.date_range("2020-01-01", periods=6)) * 2, "y": range(12)})
    d = PanelWindowDataset(f, 3, 2); assert len(d) == 4; assert d[0]["unique_id"] == "a" and d[-1]["unique_id"] == "b"


def test_moe_shapes_and_topk():
    torch.manual_seed(0); c = ForecasterConfig(lookback=5, horizon=2, d_model=8, n_heads=2, n_transformer_layers=1, latent_dim=12, expert_hidden_dim=6)
    m = MoEForecaster(2, c); y = torch.randn(3, 5); f = torch.randn(3, 5, 2); pred, diag = m(y, f)
    assert pred.shape == (3, 2); assert diag["top_k_indices"].shape == (3, 2); assert torch.allclose(diag["top_k_weights"].sum(-1), torch.ones(3))


def test_losses_and_diagnostics():
    p = torch.tensor([[.95, .03, .02], [.95, .03, .02]]); i = torch.tensor([[0], [0]])
    out = total_loss(torch.ones(2, 2), torch.zeros(2, 2), {"probabilities": p, "top_k_indices": i})
    assert set(out) == {"forecast_loss", "routing_balance_loss", "total_loss"}; assert routing_diagnostics({"probabilities": p, "top_k_indices": i})["warning"]
