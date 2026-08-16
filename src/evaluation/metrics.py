import torch


def _reduce(x, reduction): return x.mean() if reduction == "global" else x.mean(0) if reduction == "per_horizon" else x
def mae(actual, predicted, reduction="global"): return _reduce((actual - predicted).abs(), reduction)
def rmse(actual, predicted, reduction="global"): return _reduce((actual - predicted).square().mean(-1).sqrt(), reduction)
def wape(actual, predicted, reduction="global"):
    num = (actual - predicted).abs(); den = actual.abs().sum(-1).clamp_min(torch.finfo(actual.dtype).eps)
    return _reduce(num.sum(-1) / den, reduction)
def smape(actual, predicted, reduction="global"):
    value = 2 * (actual - predicted).abs() / (actual.abs() + predicted.abs()).clamp_min(torch.finfo(actual.dtype).eps)
    return _reduce(value.mean(-1), reduction)
