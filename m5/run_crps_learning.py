#!/usr/bin/env python3
"""M5 application: CRPS learning (Berrisch and Ziel, 2023) with the R package profoc.

One weight vector per quantile level is shared by all series.  The smoothing
parameter and forgetting rate are selected by CRPS on the completed validation
origins; the quantile scores of origin r update the weights only at origins o
with r + 27 < o.

  --panel full             758 food series (main text)
  --panel stable_reversal  45-series stable-reversal subset (supplementary materials)
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import regional_pool as base  # noqa: E402
import update_timing as timing  # noqa: E402
from quantile_average import inverse  # noqa: E402
from records import HORIZON, TRAIN_ORIGIN_COUNT, VALIDATION_ORIGIN_COUNT, origins  # noqa: E402
from run_m5 import TAU, panel_records  # noqa: E402

METHOD = "CRPS learning"
PROBS = np.arange(1, 100) / 100
ANCHORS = np.r_[0.0, PROBS, 1.0]
LAMBDAS = [-np.inf] + [2.0**v for v in range(-4, 14)] + [2.0**30]
FORGET = [0.0, 0.01, 0.05]


def run_r(work: Path, mode: str, r_library: Path | None) -> None:
    env = os.environ.copy()
    env["LC_ALL"] = "en_US.UTF-8"
    subprocess.run(
        ["Rscript", str(ROOT / "src/crps_learning.R"), str(work), mode,
         str(r_library) if r_library else "", str(ROOT / "src/crps_learning_engine.R")],
        check=True,
        env=env,
    )


def score(x, grids, y, c_low, c_high, weights, origin_indices):
    """Test or validation scores of the weighted quantile average."""

    rows = []
    for offset, j in enumerate(origin_indices):
        q = np.sort(np.einsum("npk,pk->np", x[j][:, 1:-1], weights[offset]), axis=1)
        for i in range(x[j].shape[0]):
            grid = grids[j, i]
            qs = np.r_[grid[0], q[i], grid[-1]]
            cdf = np.interp(grid, qs, ANCHORS, left=0.0, right=1.0)
            losses = base.score_losses(grid, cdf, y[j, i], c_low[j, i], c_high[j, i])
            q95 = float(qs[95])
            rows.append(
                {
                    "origin_index": j,
                    "series_index": i,
                    "q95": q95,
                    "actual_D28": y[j, i],
                    "crps": losses[0],
                    "twcrps_u": losses[3],
                    "newsvendor_cost": base.newsvendor_cost(q95, y[j, i], TAU),
                    "q95_coverage_point": float(y[j, i] <= q95),
                    "q95_bias_point": q95 - y[j, i],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel", choices=["full", "stable_reversal"], default="full")
    parser.add_argument("--config", type=Path, default=ROOT / "config/m5.json")
    parser.add_argument("--r-library", type=Path, help="Extra R library folder containing profoc 1.3.4")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    out = args.output_dir.expanduser().resolve()
    work = out / "crps_learning_work"
    work.mkdir(parents=True, exist_ok=True)
    records = panel_records(args.data_dir.expanduser().resolve(), config, args.panel, out)
    all_origins = origins()
    series = sorted({rec.series_id for rec in records})
    series_index = {sid: i for i, sid in enumerate(series)}
    origin_index = {day: j for j, day in enumerate(all_origins)}
    t, n, k = len(all_origins), len(series), len(config["components"])

    # Component quantiles at 0, 0.01, ..., 0.99, 1 and the scoring inputs of every record.
    x = np.empty((t, n, len(ANCHORS), k))
    grids = np.empty((t, n, int(config["grid_size"])))
    y, c_low, c_high = np.empty((t, n)), np.empty((t, n)), np.empty((t, n))
    for rec in records:
        j, i = origin_index[rec.day], series_index[rec.series_id]
        x[j, i] = np.array([inverse(rec.grid, cdf, ANCHORS) for cdf in rec.comp_cdfs]).T
        grids[j, i], y[j, i], c_low[j, i], c_high[j, i] = rec.grid, rec.y, rec.c_low, rec.c_high
    n_fit = TRAIN_ORIGIN_COUNT + VALIDATION_ORIGIN_COUNT
    for j in range(t):
        x[j].ravel(order="F").tofile(work / f"x_{j:03d}.bin")
    y[:n_fit].T.ravel(order="F").tofile(work / "y_validation.bin")
    (work / "metadata.json").write_text(
        json.dumps({"models": list(config["components"]), "n_series": n, "origins": all_origins,
                    "horizon": HORIZON, "n_fit": n_fit})
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": f"boa_{j:02d}_{f:02d}", "lambda": lam, "forget": forget}
            for j, lam in enumerate(LAMBDAS)
            for f, forget in enumerate(FORGET)
        ]
    )
    candidates.to_csv(work / "candidates.csv", index=False)

    # Validation origins whose outcomes are complete before the first test origin.
    first_test = all_origins[n_fit]
    tuning_origins = [
        j for j in range(TRAIN_ORIGIN_COUNT, n_fit)
        if timing.is_outcome_available(all_origins[j], first_test, HORIZON)
    ]
    run_r(work, "validation", args.r_library)
    rows = []
    for spec in candidates.to_dict("records"):
        w = np.fromfile(work / f"{spec['candidate_id']}_validation_weights.bin").reshape(n_fit, len(PROBS), k, order="F")
        detail = score(x, grids, y, c_low, c_high, w[tuning_origins], tuning_origins)
        rows.append({**spec, "validation_CRPS": float(detail["crps"].mean())})
    tuning = pd.DataFrame(rows).sort_values(["validation_CRPS", "candidate_id"])
    tuning.to_csv(out / "crps_learning_tuning.csv", index=False)
    selected = tuning.iloc[0].to_dict()
    if not np.isfinite(selected["lambda"]):
        selected["lambda"] = "-Inf"
    (work / "selection.json").write_text(json.dumps(selected))
    print(f"Selected: lambda={selected['lambda']}, forget={selected['forget']}", flush=True)

    # Test outcomes are passed to the learner only after the selection is saved.
    y.T.ravel(order="F").tofile(work / "y_all.bin")
    run_r(work, "test", args.r_library)
    w = np.fromfile(work / f"{selected['candidate_id']}_test_weights.bin").reshape(t - n_fit, len(PROBS), k, order="F")
    detail = score(x, grids, y, c_low, c_high, w, range(n_fit, t))
    detail.insert(0, "series_id", [series[i] for i in detail.pop("series_index")])
    detail.insert(0, "origin", [all_origins[j] for j in detail.pop("origin_index")])
    detail.insert(0, "method", METHOD)
    detail.to_csv(out / "crps_learning_predictions.csv", index=False)
    metrics = {
        "method": METHOD,
        "Cost": detail["newsvendor_cost"].mean(),
        "CRPS": detail["crps"].mean(),
        "twCRPS_U": detail["twcrps_u"].mean(),
        "q95_coverage": detail["q95_coverage_point"].mean(),
        "q95_bias": detail["q95_bias_point"].mean(),
    }
    pd.DataFrame([metrics]).to_csv(out / "crps_learning_metrics.csv", index=False)
    np.save(out / "crps_learning_weights.npy", w)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
