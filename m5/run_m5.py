#!/usr/bin/env python3
"""M5 application: forecast combinations for 28-day newsvendor orders.

Combinations are fitted on the training origins, tuned on the completed
validation origins and evaluated on the test origins.  Weights are updated only
with 28-day demands that are fully observed at the forecast origin.

  --panel full             758 food series (main text)
  --panel stable_reversal  45-series stable-reversal subset (supplementary materials)

CRPS learning is run separately by ``run_crps_learning.py``.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import benchmarks  # noqa: E402
import regional_pool as base  # noqa: E402
import update_timing as timing  # noqa: E402
from combinations import (  # noqa: E402
    evaluate_constant_pool,
    evaluate_core_delayed_trace,
    evaluate_equal_pool_baseline,
    evaluate_particle_trace,
    evaluate_recursive_gdfc_trace,
    output_predictions,
    tune_core_method,
    tune_particle_spec_full,
)
from records import (  # noqa: E402
    HORIZON,
    TRAIN_ORIGIN_COUNT,
    VALIDATION_ORIGIN_COUNT,
    LeadTimeForecasts,
    build_records,
    origins,
    quantile_maps,
    split_records,
)
from stable_reversal import select_stable_reversals  # noqa: E402

TAU = 0.95


def load_records(data_dir: Path, config: dict) -> list[base.Record]:
    """Build one record (outcome and five component distributions) per series and origin."""

    sales = pd.read_csv(data_dir / config["sales_file"])
    frames = {}
    for model in config["components"]:
        frame = pd.read_csv(data_dir / "components" / f"{model}.csv")
        frames[model] = frame.sort_values(["series_id", "origin"]).reset_index(drop=True)
    forecasts = LeadTimeForecasts(origins=origins(), horizon_days=HORIZON, model_quantiles=quantile_maps(frames))
    print(f"Building records for {len(sales)} series x {len(origins())} origins", flush=True)
    return build_records(sales, forecasts, TAU, int(config["grid_size"]))


def panel_records(data_dir: Path, config: dict, panel: str, out: Path) -> list[base.Record]:
    """Records of the full panel or of the training-selected stable-reversal subset."""

    records = load_records(data_dir, config)
    if panel == "stable_reversal":
        all_origins = origins()
        selection = select_stable_reversals(
            records, list(config["components"]), set(all_origins[:TRAIN_ORIGIN_COUNT]),
            all_origins[TRAIN_ORIGIN_COUNT], HORIZON,
        )
        selection.to_csv(out / "stable_reversal_selection.csv", index=False)
        selected = set(selection.loc[selection.stable_reversal, "series_id"])
        records = [rec for rec in records if rec.series_id in selected]
        print(f"Stable-reversal subset: {len(selected)} series", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with the sales panel and components/<model>.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel", choices=["full", "stable_reversal"], default="full")
    parser.add_argument("--config", type=Path, default=ROOT / "config/m5.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_origins = origins()
    records = panel_records(args.data_dir.expanduser().resolve(), config, args.panel, out)
    models = list(config["components"])
    train, val, test = split_records(records, all_origins)
    first_validation = all_origins[TRAIN_ORIGIN_COUNT]
    first_test = all_origins[TRAIN_ORIGIN_COUNT + VALIDATION_ORIGIN_COUNT]
    # Only outcomes fully observed at the first validation (test) origin may be used.
    train_fit = [rec for rec in train if timing.is_outcome_available(rec.day, first_validation, HORIZON)]
    train_pending = [rec for rec in train if not timing.is_outcome_available(rec.day, first_validation, HORIZON)]
    val_tuning = [rec for rec in val if timing.is_outcome_available(rec.day, first_test, HORIZON)]
    baseline = evaluate_equal_pool_baseline(val_tuning)

    metric_rows, details, weights, tuning = [], [], [], []
    selected: dict[str, dict] = {}

    def keep(metrics, detail, weight_frame, label):
        metric_rows.append(metrics)
        detail["method"] = label
        detail["split"] = "test"
        details.append(detail)
        weights.append(weight_frame)
        print(f"{label}: cost={metrics['Cost']:.3f} CRPS={metrics['CRPS']:.3f}", flush=True)

    # Score-driven core methods.
    for label, code in [
        ("Equal pool", "equal"),
        ("Global GAS", "dynamic_global"),
        ("Static RALP", "static_ralp"),
        ("SD-RALP", "sd_ralp"),
    ]:
        if code == "equal":
            static = base.fit_static_weights(train, config["empirical_run"]["eta_grid"][0])
            alpha = beta = rho = 0.0
        else:
            chosen, _, rows = tune_core_method(train_fit, train_pending, val_tuning, code, config, HORIZON, baseline)
            rows["method"] = label
            tuning.append(rows)
            static = base.fit_static_weights(train, float(chosen["eta"]))
            alpha, beta, rho = float(chosen["alpha"]), float(chosen["beta"]), float(chosen["rho"])
            selected[label] = {key: float(chosen[key]) for key in ("eta", "alpha", "beta", "rho")}
        metrics, detail, weight_frame, _ = evaluate_core_delayed_trace(
            test, code, label, "test", static, alpha, beta, rho, HORIZON,
            warm_records=val, enqueue_score_updates=True,
        )
        keep(metrics, detail, weight_frame, label)

    # Static density pools fitted on the training outcomes.
    equal_weights = np.repeat(1.0 / len(models), len(models))
    for label, pool_weights, forecast in [
        ("Log-score pool", benchmarks.fit_log_score_pool(train)[0], None),
        ("BMA", benchmarks.fit_bma_weights(train), None),
        ("Quantile-average pool", equal_weights, benchmarks.quantile_average_pool),
    ]:
        metrics, detail, weight_frame = evaluate_constant_pool(test, label, "test", pool_weights, forecast)
        keep(metrics, detail, weight_frame, label)

    # Recursive GDFC, refitted at every origin on all completed outcomes.
    label = "Recursive GDFC"
    metrics, detail, weight_frame, _, _ = evaluate_recursive_gdfc_trace(
        train + val, test, "test", HORIZON, enqueue_score_updates=True
    )
    keep(metrics, detail, weight_frame, label)

    # Time-varying weights by particle filtering.
    for label, diversity in [("TVW", False), ("DTVW", True)]:
        spec, rows = tune_particle_spec_full(train, val_tuning, config, diversity, HORIZON, baseline)
        rows["method"] = label
        tuning.append(rows)
        selected[label] = asdict(spec)
        metrics, detail, weight_frame, _ = evaluate_particle_trace(
            train, val, test, "test", spec, diversity, HORIZON, enqueue_score_updates=True
        )
        keep(metrics, detail, weight_frame, label)

    metrics = pd.DataFrame(metric_rows)
    predictions = output_predictions(details)
    weights = pd.concat(weights, ignore_index=True)
    tuning = pd.concat(tuning, ignore_index=True, sort=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    tuning.to_csv(out / "tuning.csv", index=False)
    (out / "selected_parameters.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    columns = ["method", "Cost", "CRPS", "twCRPS_U", "q95_coverage", "q95_bias"]
    print(metrics.sort_values("Cost")[columns].to_string(index=False))


if __name__ == "__main__":
    main()
