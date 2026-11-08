# SD-RALP: code for the empirical applications

Code for the two empirical applications of "Region-Adaptive Probabilistic Forecast
Combination for Risk-Sensitive Operational Decisions":

| Folder | Application | Results |
|---|---|---|
| [`electricity/`](electricity/README.md) | Shanxi day-ahead prices and battery scheduling | Section 5.1, Table 5 |
| [`m5/`](m5/README.md) | M5 retail data and newsvendor replenishment | Section 5.2, Table 6; supplementary Table S1 |

Each folder runs on its own. Both start from the component forecasts: they fit, tune and
evaluate the forecast combinations and score the resulting decisions.

The combinations are SD-RALP, Static RALP, Global GAS, Equal pool, Log-score pool, BMA,
Quantile-average pool, TVW, DTVW, Recursive GDFC and CRPS learning.

## Requirements

- Python 3.10 or later with numpy, pandas and scipy (tested with Python 3.14.4,
  numpy 2.4.6, pandas 3.0.3, scipy 1.17.1).
- For CRPS learning only: R with the packages `profoc` (version 1.3.4) and `jsonlite`
  (tested with R 4.5.0).

```bash
pip install -r electricity/requirements.txt
Rscript -e 'remotes::install_version("profoc", "1.3.4"); install.packages("jsonlite")'
```
