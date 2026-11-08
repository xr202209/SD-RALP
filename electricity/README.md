# Electricity application (Section 5.1)

Density combinations of five component forecasts of Shanxi day-ahead prices, evaluated by
forecast scores and by the decision loss of a risk-averse battery schedule.

## Input data

`--data-dir` must contain

- `shanxi_electricity.csv`: quarter-hourly prices with columns `datetime` (end of the
  quarter-hour, so `00:00` belongs to the previous day) and `day_ahead_price`, from
  2 May 2024 to at least 30 August 2025;
- `components/<model>.npz`: one file per component, for the 420 forecast days from
  7 July 2024 to 30 August 2025, with arrays
  - `delivery_days` (420), `fit_end_by_day` (420): forecast day and last day used to fit it,
  - `quantile_levels` (99): 0.01, ..., 0.99,
  - `quantiles` (420, 96, 99) and `p_zero` (420, 96): quantiles and probability of a zero price,
  - optionally `pmf` (420, 96, 301) and `pmf_grid` (301): probability masses on the grid
    0, 5, ..., 1500, used instead of the quantiles when present.

| File name | Component in the paper |
|---|---|
| `LeafWeightedQRF.npz` | QRF |
| `SpikeHurdleQRF.npz` | Hurdle |
| `LowerInterceptSpikeQRF90.npz` | LowCal |
| `UpperQ95TimeSpikeW60C001.npz` | PeakCal |
| `SplineGAMEVT.npz` | GAM-EVT |

The sample split, regions, validation criterion, grids and battery settings are in
`config/electricity.json`.

## Running

```bash
python run_electricity.py tune     --data-dir DATA --output-dir OUT   # validation search
python run_electricity.py forecast --data-dir DATA --output-dir OUT   # test forecasts
python run_crps_learning.py        --data-dir DATA --output-dir OUT   # CRPS learning (R)
python run_electricity.py battery  --data-dir DATA --output-dir OUT   # battery schedules
```

`tune` writes the selected hyperparameters to `OUT/selected_parameters.json`.
`forecast` uses `config/selected_parameters.json`, the values selected for the paper,
unless `--parameters` points to another file. `battery` evaluates every forecast file in
`OUT/forecasts/`, so run it after `forecast` and `run_crps_learning.py`.
`run_crps_learning.py` accepts `--r-library` for an R library folder containing profoc.

On one core of an Apple M-series laptop the stages take about 15 minutes (tune),
1 minute (forecast), 4 minutes (CRPS learning) and 7 minutes (battery).

## Output

| File | Content | Table 5 column |
|---|---|---|
| `forecasts/<method>.npz` | test-day probability masses and weights | |
| `forecasts/forecast_metrics.csv` | `CRPS`, `twCRPS_L`, `twCRPS_U` | CRPS, twCRPS$_L$, twCRPS$_U$ |
| `crps_learning_metrics.csv` | the same scores for CRPS learning | |
| `battery/battery_metrics.csv` | `mean_regret`, `mean_realised_profit`, `profit_capture_ratio` | Decision loss, Realised profit, Profit capture |
| `battery/battery_daily.csv` | daily schedule results | |

## Files

- `run_electricity.py`: tuning, test forecasts and battery evaluation.
- `run_crps_learning.py`, `src/crps_learning.R`, `src/crps_learning_engine.R`: CRPS learning.
- `src/data.py`: price data, component forecasts, forecast scores and validation criterion.
- `src/combination.py`: SD-RALP, Static RALP, Global GAS, Equal pool, Log-score pool, BMA,
  Quantile-average pool, TVW and DTVW.
- `src/gdfc.py`: Recursive GDFC.
- `src/distributions.py`: discrete distributions on the price grid and their scores.
- `src/battery.py`: scenario copula and the mean-CVaR battery linear program.
