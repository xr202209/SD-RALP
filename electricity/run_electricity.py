#!/usr/bin/env python3
"""Electricity application: forecast combinations and battery scheduling.

Stages
  tune      select hyperparameters on the validation days
  forecast  produce the combined forecasts for the test days
  battery   evaluate the test forecasts through the risk-averse battery schedule

CRPS learning is run separately by ``run_crps_learning.py``; its forecasts are
written to the same forecast directory and evaluated by the ``battery`` stage.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from battery import (  # noqa: E402
    BatterySpec,
    aggregate_results,
    build_battery_matrices,
    build_checkerboard_copulas,
    pmf_to_scenarios,
    realised_schedule_metrics,
    solve_perfect_information,
    solve_risk_averse_schedule,
)
from combination import (  # noqa: E402
    ParticleSpec,
    StaticWeights,
    compute_regional_stats,
    concatenate_panels,
    fit_bma_weights,
    fit_log_score_pool,
    fit_static_weights,
    run_constant_pool,
    run_core_method,
    run_particle_pool,
    run_quantile_average_pool,
    score_combination,
    tune_core_method,
    tune_particle_pool,
)
from data import (  # noqa: E402
    forecast_days,
    forecast_scores,
    load_component_panel,
    load_price_curves,
    particle_specs,
    read_json,
    refined_values,
    slice_panel,
    validation_objective,
    write_json,
)
from gdfc import recursive_kmpf, wrap_kmpf_result  # noqa: E402

METHODS = {
    "sd_ralp": "SD-RALP",
    "static_ralp": "Static RALP",
    "global_gas": "Global GAS",
    "equal": "Equal pool",
    "log_score": "Log-score pool",
    "bma": "BMA",
    "quantile_average": "Quantile-average pool",
    "tvw": "TVW",
    "dtvw": "DTVW",
    "recursive_gdfc": "Recursive GDFC",
}
# Internal names of the score-driven core methods.
CORE_CODE = {"sd_ralp": "sd_ralp", "static_ralp": "static_ralp", "global_gas": "dynamic_global"}


def load_inputs(config: dict, data_dir: Path):
    curves = load_price_curves(data_dir / config["price_file"], config["target"], config["split"]["test_end"])
    panel = load_component_panel(config, data_dir / "components", curves)
    days = forecast_days(config["split"])
    n_train = int((days <= config["split"]["train_end"]).sum())
    n_validation = int((days <= config["split"]["validation_end"]).sum())
    train = slice_panel(panel, 0, n_train)
    validation = slice_panel(panel, n_train, n_validation)
    test = slice_panel(panel, n_validation, panel.days)
    return curves, days[n_validation:], train, validation, test


def tune(config: dict, train, validation, out: Path) -> None:
    equal = run_core_method(validation, "equal", StaticWeights.equal(train.model_count))
    objective = validation_objective(config["validation_objective"], score_combination(validation, equal))
    train_stats = compute_regional_stats(train)
    validation_stats = compute_regional_stats(validation)
    parameters: dict[str, dict] = {}
    rows: list[dict] = []

    for method in ("global_gas", "static_ralp", "sd_ralp"):
        print(f"Tuning {METHODS[method]}", flush=True)
        tuned = tune_core_method(
            train,
            validation,
            CORE_CODE[method],
            objective=objective,
            train_stats=train_stats,
            validation_stats=validation_stats,
            **config["core_tuning"],
        )
        parameters[method] = {key: float(tuned.parameters[key]) for key in ("eta", "alpha", "beta", "rho")}
        rows += [{"method": METHODS[method], **row} for row in tuned.rows]

    pf = config["particle_filter"]
    for method, diversity in (("tvw", False), ("dtvw", True)):
        print(f"Tuning {METHODS[method]}", flush=True)
        coarse = tune_particle_pool(
            train,
            validation,
            particle_specs(pf, pf["alpha1_grid"], pf["alpha2_grid"] if diversity else [0.0]),
            use_diversity=diversity,
            objective=objective,
        )
        fine = tune_particle_pool(
            train,
            validation,
            particle_specs(
                pf,
                refined_values(coarse.spec.alpha1, pf["refine_radius"], pf["refine_step"]),
                refined_values(coarse.spec.alpha2, pf["refine_radius"], pf["refine_step"]) if diversity else [0.0],
            ),
            use_diversity=diversity,
            objective=objective,
        )
        parameters[method] = asdict(fine.spec)
        rows += [{"method": METHODS[method], "stage": "coarse", **row} for row in coarse.rows]
        rows += [{"method": METHODS[method], "stage": "refined", **row} for row in fine.rows]

    gdfc = config["recursive_gdfc"]
    print("Tuning Recursive GDFC", flush=True)
    scores = []
    for p in gdfc["p_grid"]:
        score, *_ = recursive_kmpf(
            train,
            validation,
            p=int(p),
            refit_days=int(gdfc["refit_days"]),
            n_starts=int(gdfc["n_starts"]),
            max_iter=int(gdfc["max_iter"]),
            random_seed=int(gdfc["random_seed"]),
            collect_pmf=False,
            min_probability=float(gdfc["min_probability"]),
        )
        scores.append((score, int(p)))
        rows.append({"method": METHODS["recursive_gdfc"], "p": int(p), "validation_Log_score": float(score)})
    parameters["recursive_gdfc"] = {"p": min(scores)[1]}

    write_json(out / "selected_parameters.json", parameters)
    pd.DataFrame(rows).to_csv(out / "tuning.csv", index=False)
    print(f"Selected parameters written to {out / 'selected_parameters.json'}", flush=True)


def run_method(method: str, config: dict, parameters: dict, train, validation, test):
    """Test-period forecasts of one method, fitted on training and updated sequentially."""

    warm = concatenate_panels([train, validation])
    if method == "equal":
        return run_core_method(test, "equal", StaticWeights.equal(train.model_count))
    if method in CORE_CODE:
        params = parameters[method]
        static = fit_static_weights(train, float(params["eta"]))
        dynamic = method in ("sd_ralp", "global_gas")
        return run_core_method(
            test,
            CORE_CODE[method],
            static,
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            rho=float(params["rho"]),
            warm_data=warm if dynamic else None,
        )
    if method == "log_score":
        return run_constant_pool(test, fit_log_score_pool(train)[0], method=METHODS[method])
    if method == "bma":
        return run_constant_pool(test, fit_bma_weights(train), method=METHODS[method])
    if method == "quantile_average":
        return run_quantile_average_pool(test, quantile_points=int(config["quantile_average"]["quantile_points"]))
    if method in ("tvw", "dtvw"):
        spec = ParticleSpec(**parameters[method])
        return run_particle_pool(train, test, spec, use_diversity=method == "dtvw", warm_data=validation)
    if method == "recursive_gdfc":
        gdfc = config["recursive_gdfc"]
        _, pmf, _, _ = recursive_kmpf(
            warm,
            test,
            p=int(parameters[method]["p"]),
            refit_days=int(gdfc["refit_days"]),
            n_starts=int(gdfc["n_starts"]),
            max_iter=int(gdfc["max_iter"]),
            random_seed=int(gdfc["random_seed"]) + 500_000,
            collect_pmf=True,
            min_probability=float(gdfc["min_probability"]),
            evaluate_log_score=False,
        )
        return wrap_kmpf_result(test, pmf)
    raise ValueError(method)


def forecast(config: dict, parameters: dict, methods: list[str], days, train, validation, test, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows, daily = [], []
    for method in methods:
        print(f"Forecasting {METHODS[method]}", flush=True)
        result = run_method(method, config, parameters, train, validation, test)
        np.savez_compressed(
            out / f"{method}.npz",
            method=np.asarray(METHODS[method]),
            pmf=result.pmf.astype(np.float32),
            grid=test.grid,
            delivery_days=days.to_numpy(dtype="datetime64[D]"),
            weights=np.asarray(result.weights_trace, dtype=np.float32),
        )
        row, day_scores = forecast_scores(METHODS[method], days, test, result)
        rows.append(row)
        daily.append(day_scores)
    pd.DataFrame(rows).to_csv(out / "forecast_metrics.csv", index=False)
    pd.concat(daily).to_csv(out / "forecast_daily.csv", index=False)


def battery(config: dict, curves: pd.DataFrame, forecast_dir: Path, out: Path, max_days: int | None) -> None:
    spec = BatterySpec(**config["battery"])
    spec.validate()
    files = sorted(forecast_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no forecasts in {forecast_dir}")
    with np.load(files[0]) as payload:
        days = payload["delivery_days"].astype("datetime64[D]")
    if max_days is not None:
        days = days[:max_days]
    uniforms, copula = build_checkerboard_copulas(
        curves, days, spec.copula_history_days, spec.copula_replications, spec.copula_random_seed
    )
    actual = curves.loc[pd.DatetimeIndex(days.astype("datetime64[ns]"))].to_numpy(float)
    matrices = build_battery_matrices(spec)
    perfect = [solve_perfect_information(y, spec, matrices) for y in actual]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        with np.load(path) as payload:
            method = str(payload["method"])
            pmf = payload["pmf"].astype(float)
            grid = payload["grid"].astype(float)
            if not np.array_equal(payload["delivery_days"].astype("datetime64[D]")[: len(days)], days):
                raise RuntimeError(f"{path.name}: delivery days differ")
        print(f"Battery: {method}", flush=True)
        for i, day in enumerate(days):
            scenarios = pmf_to_scenarios(pmf[i], grid, uniforms[i])
            solution = solve_risk_averse_schedule(scenarios, spec, matrices)
            rows.append(
                {
                    "method": method,
                    "source_type": "combination",
                    "split": "test",
                    "delivery_day": str(day),
                    "forecast_objective": solution["forecast_objective"],
                    "common_uniform_sha256": copula[i]["uniform_sha256"],
                    **realised_schedule_metrics(solution, actual[i], perfect[i], spec),
                }
            )
    daily = pd.DataFrame(rows)
    daily.to_csv(out / "battery_daily.csv", index=False)
    aggregate_results(daily).to_csv(out / "battery_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["tune", "forecast", "battery"])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with the price file and components/<model>.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/electricity.json")
    parser.add_argument("--parameters", type=Path, default=ROOT / "config/selected_parameters.json",
                        help="Selected hyperparameters used by the forecast stage")
    parser.add_argument("--method", choices=list(METHODS), action="append",
                        help="Forecast only this method (repeatable); default all")
    parser.add_argument("--max-days", type=int, help="Battery stage: evaluate only the first days (quick check)")
    args = parser.parse_args()

    config = read_json(args.config)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    curves, days, train, validation, test = load_inputs(config, args.data_dir.expanduser().resolve())
    if args.stage == "tune":
        tune(config, train, validation, out)
    elif args.stage == "forecast":
        forecast(config, read_json(args.parameters), args.method or list(METHODS), days, train, validation, test,
                 out / "forecasts")
    else:
        battery(config, curves, out / "forecasts", out / "battery", args.max_days)


if __name__ == "__main__":
    main()
