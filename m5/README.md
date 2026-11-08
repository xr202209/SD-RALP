# M5 application (Section 5.2)

Combinations of five component forecasts of 28-day demand for 758 Walmart food series,
evaluated by forecast scores and the newsvendor cost of the 0.95-quantile order. Weights
are fitted on the 30 training origins, hyperparameters are selected on the 37 validation
origins whose demands are complete before testing, and results are reported for the 61
test origins. A 28-day demand updates the weights only once it is fully observed.

## Input data

`--data-dir` must contain

- `m5_food_sales_758.csv`: rows of `sales_train_evaluation.csv` from the M5 competition
  for the 758 selected series (columns `id`, `item_id`, `dept_id`, `cat_id`, `store_id`,
  `state_id`, `d_1`, ..., `d_1941`). A FOODS series is selected if it has sales on at
  least 80% of days 1-999 and on at least 85% of days 635-999.
- `components/<model>.csv`: one file per component with columns `series_id`, `origin`
  and the quantiles `q10`, `q50`, `q90`, `q95`, `q99` of the 28-day demand, for the 131
  weekly origins `d_1004`, `d_1011`, ..., `d_1914`.

| File name | Component in the paper |
|---|---|
| `LightGBMQuantile.csv` | LightGBM |
| `DeepAR.csv` | DeepAR |
| `KNNAnalogue.csv` | KNN analogue |
| `GAMEVT.csv` | GAM-EVT |
| `LocalGAMQuantile.csv` | Local GAM-QR |

Grids and the validation search are in `config/m5.json`.

## Running

```bash
python run_m5.py            --data-dir DATA --output-dir OUT/full
python run_crps_learning.py --data-dir DATA --output-dir OUT/full
python run_m5.py            --data-dir DATA --output-dir OUT/stable --panel stable_reversal
python run_crps_learning.py --data-dir DATA --output-dir OUT/stable --panel stable_reversal
```

`--panel full` (default) gives Table 6. `--panel stable_reversal` selects the 45-series
subset from training outcomes and refits all combinations on it (supplementary Table S1).
`run_crps_learning.py` accepts `--r-library` for an R library folder containing profoc.

On one core of an Apple M-series laptop the full panel takes about 5 minutes
(`run_m5.py`) and 3 minutes (CRPS learning); the subset about 1 minute each.

## Output

| File | Content |
|---|---|
| `metrics.csv` | test results of ten combinations |
| `crps_learning_metrics.csv` | test results of CRPS learning |
| `predictions.csv`, `crps_learning_predictions.csv` | one row per test forecast |
| `weights.csv` | component weights per origin and region |
| `tuning.csv`, `crps_learning_tuning.csv`, `selected_parameters.json` | validation search |
| `stable_reversal_selection.csv` | subset selection (stable-reversal panel only) |

Columns of `metrics.csv` used in Table 6: `Cost` (decision loss), `CRPS`, `twCRPS_U`,
`q95_coverage` (Coverage$_{0.95}$) and `q95_bias` (order bias).

## Files

- `run_m5.py`: fitting, validation search and test evaluation of ten combinations.
- `run_crps_learning.py`, `src/crps_learning.R`, `src/crps_learning_engine.R`: CRPS learning.
- `src/records.py`: forecast origins, component distributions and outcomes.
- `src/regional_pool.py`: scores, newsvendor cost, static weights, Global GAS and SD-RALP updates.
- `src/combinations.py`: evaluation of each method with delayed outcomes; validation criterion.
- `src/benchmarks.py`: Log-score pool, BMA, Recursive GDFC, TVW and DTVW.
- `src/quantile_average.py`: Quantile-average pool.
- `src/tuning_batch.py`: batched validation scoring.
- `src/update_timing.py`: availability of 28-day outcomes.
- `src/stable_reversal.py`: stable-reversal subset.
