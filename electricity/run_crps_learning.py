#!/usr/bin/env python3
"""Electricity application: CRPS learning (Berrisch and Ziel, 2023) with the R package profoc.

The smoothing parameter and forgetting rate are selected by validation CRPS.  The
selected learner is then run through the test days, updating after each delivery
day.  The forecasts are written to <output-dir>/forecasts/crps_learning.npz, where
``run_electricity.py battery`` evaluates them together with the other methods.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from combination import CombinationResult  # noqa: E402
from data import forecast_scores, read_json, write_json  # noqa: E402
from distributions import discrete_crps, pmf_to_quantiles, quantiles_to_discrete  # noqa: E402
from run_electricity import load_inputs  # noqa: E402

METHOD = "CRPS learning"
PROBS = np.arange(1, 100) / 100
CELL = np.diff(np.r_[0, (PROBS[:-1] + PROBS[1:]) / 2, 1])
LAMBDAS = [-np.inf] + [2.0**v for v in range(-4, 14)] + [2.0**30]
FORGET = [0.0, 0.01, 0.05]


def combined(x, w, grid):
    """Weighted quantile average mapped back to a PMF on the price grid."""

    q = np.sort(np.einsum("dhpk,dpk->dhp", x[:, :, 1:-1, :], w), axis=-1)
    # The endpoint atom at zero follows from the same probability cells.
    pzero = np.sum((q <= grid[0]) * CELL, axis=-1)
    return quantiles_to_discrete(q, PROBS, pzero, grid=grid, repair_monotonic=False)


def run_r(work: Path, mode: str, r_library: Path | None) -> None:
    env = os.environ.copy()
    env["LC_ALL"] = "en_US.UTF-8"
    subprocess.run(
        ["Rscript", str(ROOT / "src/crps_learning.R"), str(work), mode,
         str(r_library) if r_library else "", str(ROOT / "src/crps_learning_engine.R")],
        check=True,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/electricity.json")
    parser.add_argument("--r-library", type=Path, help="Extra R library folder containing profoc 1.3.4")
    args = parser.parse_args()

    config = read_json(args.config)
    out = args.output_dir.expanduser().resolve()
    work = out / "crps_learning_work"
    work.mkdir(parents=True, exist_ok=True)
    _, test_days, train, validation, test = load_inputs(config, args.data_dir.expanduser().resolve())
    grid = test.grid
    n_fit = train.days + validation.days
    component_pmf = np.concatenate([train.component_pmf, validation.component_pmf, test.component_pmf])
    actual = np.concatenate([train.actual, validation.actual, test.actual])
    days, slots, models = component_pmf.shape[0], component_pmf.shape[1], component_pmf.shape[2]

    # Component quantiles at levels 0.01, ..., 0.99, padded with the support bounds.
    x = np.empty((days, slots, len(PROBS) + 2, models))
    for d in range(days):
        x[d, :, 1:-1, :] = pmf_to_quantiles(component_pmf[d], PROBS, grid).transpose(0, 2, 1)
        x[d, :, 0, :] = grid[0]
        x[d, :, -1, :] = grid[-1]
        x[d].ravel(order="F").tofile(work / f"x_{d:03d}.bin")
    actual[:n_fit].T.ravel(order="F").tofile(work / "y_validation.bin")
    write_json(
        work / "metadata.json",
        # One origin per delivery day; a day's prices update the weights from the next day on.
        {"models": list(config["components"]), "n_series": slots, "origins": list(range(1, days + 1)),
         "horizon": 1, "n_fit": n_fit},
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": f"boa_{j:02d}_{f:02d}", "lambda": lam, "forget": forget}
            for j, lam in enumerate(LAMBDAS)
            for f, forget in enumerate(FORGET)
        ]
    )
    candidates.to_csv(work / "candidates.csv", index=False)

    run_r(work, "validation", args.r_library)
    rows = []
    for spec in candidates.to_dict("records"):
        w = np.fromfile(work / f"{spec['candidate_id']}_validation_weights.bin").reshape(
            n_fit, len(PROBS), models, order="F"
        )
        dist = combined(x[train.days:n_fit], w[train.days:n_fit], grid)
        crps = discrete_crps(dist.pmf, validation.actual, grid, reduction="mean")
        rows.append({**spec, "validation_CRPS": float(crps)})
    tuning = pd.DataFrame(rows).sort_values(["validation_CRPS", "candidate_id"])
    tuning.to_csv(out / "crps_learning_tuning.csv", index=False)
    selected = tuning.iloc[0].to_dict()
    if not np.isfinite(selected["lambda"]):
        selected["lambda"] = "-Inf"
    write_json(work / "selection.json", selected)
    print(f"Selected: lambda={selected['lambda']}, forget={selected['forget']}", flush=True)

    # Test outcomes are passed to the learner only after the selection is saved.
    actual.T.ravel(order="F").tofile(work / "y_all.bin")
    run_r(work, "test", args.r_library)
    w = np.fromfile(work / f"{selected['candidate_id']}_test_weights.bin").reshape(
        test.days, len(PROBS), models, order="F"
    )
    dist = combined(x[n_fit:], w, grid)
    result = CombinationResult(
        METHOD, dist.pmf, dist.cdf, np.repeat(w.mean(axis=1)[:, None, :], 3, axis=1), np.ones((test.days, slots))
    )
    result.validate(test)
    forecasts = out / "forecasts"
    forecasts.mkdir(exist_ok=True)
    np.savez_compressed(
        forecasts / "crps_learning.npz",
        method=np.asarray(METHOD),
        pmf=dist.pmf.astype(np.float32),
        grid=grid,
        delivery_days=test_days.to_numpy(dtype="datetime64[D]"),
        weights=w.astype(np.float32),
    )
    row, daily = forecast_scores(METHOD, test_days, test, result)
    pd.DataFrame([row]).to_csv(out / "crps_learning_metrics.csv", index=False)
    daily.to_csv(out / "crps_learning_daily.csv", index=False)
    print(f"Test CRPS: {row['CRPS']:.3f}", flush=True)


if __name__ == "__main__":
    main()
