# Retail Demand Forecasting: Transformer-MoE vs ETS

This repository evaluates whether a Transformer Mixture-of-Experts (MoE) model can outperform a classical ETS forecast on panel retail demand data.

The main finding is that ETS is the stronger model for the current dataset and evaluation period. The MoE is retained as a neural challenger, but its additional complexity did not produce better aggregate accuracy.

## Experiment

Input data uses the panel format:

```text
unique_id, ds, y
```

Each `unique_id` is a demand series, `ds` is the observation date, and `y` is demand. The main experiment uses a 104-day lookback and predicts a 13-day horizon.

### ETS baseline

The statistical baseline is additive Holt-Winters exponential smoothing with additive trend, damped trend, additive weekly seasonality, and a 13-day forecast horizon.

### Transformer-MoE challenger

The neural model uses per-window target normalization, a temporal Transformer encoder, sinusoidal positional encoding, calendar features, optional target lags (`7, 14, 28`), a learned series embedding, and a top-2 router over four MLP experts. It produces a weighted direct 13-step forecast.

The implementation also passes known future calendar features for each forecast date, including weekday, week, month, day-of-year, and time-position features.

## Reproduce the comparison

Generate the ETS backtest:

```bash
python scripts/run_ets_backtest.py \
  --data m5_ca3_sub_dropped.csv \
  --series-fraction 0.10 \
  --backtest-cutoff 2016-05-15 \
  --output m5_ca3_ets_backtest.csv
```

Train and evaluate the MoE:

```bash
python scripts/train_smoke.py \
  --data m5_ca3_sub_dropped.csv \
  --series-fraction 0.10 \
  --lookback 104 \
  --lags 7,14,28 \
  --backtest-cutoff 2016-05-15 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --seed 7 \
  --device mps \
  --forecast-output m5_ca3_backtest_104d_future_calendar.csv
```

Compare the forecasts:

```bash
python scripts/compare_backtests.py \
  --moe m5_ca3_backtest_104d_future_calendar.csv \
  --naive m5_ca3_ets_backtest.csv \
  --baseline-label ets \
  --output moe_104d_future_calendar_vs_ets_comparison.csv
```

## Results

On the saved backtest with cutoff `2016-05-15`:

| Model | Absolute error | Bias | WAPE-like score | FQ |
|---|---:|---:|---:|---:|
| MoE with future calendar features | 1738.7391 | 0.029152 | 0.439297 | 0.468449 |
| ETS | 1616.8084 | -0.049391 | 0.408491 | 0.457882 |

ETS has lower absolute error and lower FQ. It also wins on the majority of evaluated series in the saved comparisons.

The comparison script’s original metric label `MAPE` is calculated as total absolute error divided by total actual demand, so it is closer to WAPE than conventional per-observation MAPE. This README calls it a WAPE-like score.

## Conclusion

For this dataset and cutoff, the Transformer-MoE does not outperform ETS. Longer lookback, target lags, alternative expert activations, scaling changes, and future calendar features did not overcome the strength of the classical seasonal baseline.

The experiment suggests that weekly seasonality and local demand dynamics explain much of the predictable signal, and that the MoE routing complexity is not currently justified by the forecast results.

This is an empirical result for this dataset and evaluation setup, not a claim that ETS is universally superior.

## Repository layout

```text
src/                               Model, data, features, losses, evaluation
scripts/train_smoke.py             MoE training and backtesting
scripts/run_ets_backtest.py        ETS backtesting
scripts/compare_backtests.py       Forecast comparison
tests/                              Core tests
```

Run the tests with:

```bash
pytest -q
```
