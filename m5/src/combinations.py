"""Combination methods of the M5 application, evaluated record by record.

Dynamic methods update their weights at an origin only with outcomes that are
fully observed by then (origin + 27 < current origin).
"""

from __future__ import annotations

from itertools import groupby
from typing import Any, Callable

import numpy as np
import pandas as pd

import benchmarks
import quantile_average
import update_timing as timing
import regional_pool as base
from tuning_batch import Panel as TuningPanel
from records import QCOLS, QUANTILE_LEVELS

GDFC_LABEL = "Recursive GDFC"
POOL_MODELS = [
    "LightGBMQuantile",
    "DeepAR",
    "KNNAnalogue",
    "GAMEVT",
    "LocalGAMQuantile",
]
TAU_STAR = 0.95
# Validation criterion: weighted scores relative to the Equal pool.
VALIDATION_WEIGHTS = {"crps": 0.05, "twcrps_u": 0.475, "qs95": 0.475}
METRIC_RENAME = {
    "crps": "CRPS",
    "twcrps_u": "twCRPS_U",
    "log_score": "Log_score",
    "tail_log_score": "Tail_LS",
    "newsvendor_cost": "Cost",
    "service_level": "Service",
    "ecal": "ECAL",
    "brier_u": "Brier_U",
    "mean_q_tau": "q95_mean",
}


def prediction_row_with_quantiles(rec: base.Record, pdf: np.ndarray, cdf: np.ndarray) -> dict[str, Any]:
    row = base.prediction_row(rec, pdf, cdf)
    for q, col in zip(QUANTILE_LEVELS, QCOLS):
        row[col] = max(0.0, base.quantile_from_cdf(rec.grid, cdf, float(q)))
    row["origin"] = int(row.pop("day"))
    row["actual_D28"] = float(row["y"])
    row["q95_bias_point"] = float(row["q95"] - row["actual_D28"])
    row["q95_coverage_point"] = float(row["actual_D28"] <= row["q95"])
    return row


def summarize_detail(detail: pd.DataFrame, records: list[base.Record], method: str, split: str, upper_weights: list[np.ndarray] | None = None) -> dict[str, Any]:
    metrics = base.summarize_predictions(detail, records, upper_weights, POOL_MODELS)
    row: dict[str, Any] = {
        "method": method,
        "split": split,
        "n_series": int(detail["series_id"].nunique()),
        "n_predictions": int(len(detail)),
    }
    for src, dst in METRIC_RENAME.items():
        row[dst] = float(metrics[src])
    row["q95_bias"] = float(detail["q95_bias_point"].mean())
    row["q95_coverage"] = float(detail["q95_coverage_point"].mean())
    row["upper_exceedance_rate"] = float((detail["actual_D28"] > detail["q90"]).mean())
    row["mean_PIT"] = float(detail["pit"].mean())
    row["CVaR_error"] = float(metrics["cvar_abs_error"])
    for name in POOL_MODELS:
        key = f"upper_w_{name}_mean"
        if key in metrics:
            row[key] = float(metrics[key])
    return row


def validation_metric_value(metrics: dict[str, float], key: str) -> float:
    if key == "qs95":
        # The quantile score at TAU_STAR is (1 - TAU_STAR) times the newsvendor cost.
        return float((1.0 - TAU_STAR) * metrics["newsvendor_cost"])
    return float(metrics[key])


def validation_objective_from_base(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> float:
    objective = 0.0
    for key, weight in VALIDATION_WEIGHTS.items():
        denominator = max(validation_metric_value(baseline_metrics, key), 1e-12)
        objective += float(weight) * validation_metric_value(metrics, key) / denominator
    return float(objective)


def evaluate_equal_pool_baseline(records: list[base.Record]) -> dict[str, float]:
    weights = np.repeat(1.0 / len(POOL_MODELS), len(POOL_MODELS))
    rows = []
    upper_weights = []
    for rec in records:
        pdf, cdf = base.combine_record(rec, weights, weights, weights)
        rows.append(base.prediction_row(rec, pdf, cdf))
        upper_weights.append(weights.copy())
    metrics = base.summarize_predictions(pd.DataFrame(rows), records, upper_weights, POOL_MODELS)
    metrics["qs95"] = validation_metric_value(metrics, "qs95")
    return metrics


def tune_core_method(
    train_fit: list[base.Record],
    warm_records: list[base.Record],
    val_tuning: list[base.Record],
    method: str,
    config: dict[str, Any],
    horizon_days: int,
    objective_baseline: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    run = config["empirical_run"]
    eta_grid = [float(x) for x in run["eta_grid"]]
    alpha_grid = [float(x) for x in run["alpha_grid"]]
    beta_grid = [float(x) for x in run["beta_grid"]]
    rho_grid = [float(x) for x in run["rho_grid"]]
    rows: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], dict[str, np.ndarray]] | None = None
    tuning_panel = TuningPanel(warm_records, val_tuning)

    for eta in eta_grid:
        static = base.fit_static_weights(train_fit, eta)
        if method == "static_ralp":
            candidates = [(0.0, 0.0, 0.0)]
        elif method == "dynamic_global":
            candidates = [(alpha, beta, 0.0) for alpha in alpha_grid for beta in beta_grid]
        elif method == "sd_ralp":
            candidates = [(alpha, beta, rho) for alpha in alpha_grid for beta in beta_grid for rho in rho_grid]
        else:
            raise ValueError(method)

        for alpha, beta, rho in candidates:
            metrics = tuning_panel.core(
                method,
                static,
                alpha,
                beta,
                rho,
                horizon_days,
            )
            objective = validation_objective_from_base(metrics, objective_baseline)
            row = {
                "method_type": "core",
                "method_code": method,
                "eta": eta,
                "alpha": alpha,
                "beta": beta,
                "rho": rho,
                "validation_objective": objective,
                "validation_CRPS": float(metrics["crps"]),
                "validation_twCRPS_U": float(metrics["twcrps_u"]),
                "validation_Brier_U": float(metrics["brier_u"]),
                "validation_QS95": validation_metric_value(metrics, "qs95"),
                "validation_ECAL": float(metrics["ecal"]),
                "validation_Cost": float(metrics["newsvendor_cost"]),
                "validation_Service": float(metrics["service_level"]),
                "n_static_fit_records": int(len(train_fit)),
                "n_validation_tuning_records": int(len(val_tuning)),
            }
            rows.append(row)
            if best is None or objective < best[0]:
                best = (objective, row, static)
        print(f"Tuned {method} eta={eta}: {len(candidates)} candidate(s)", flush=True)

    if best is None:
        raise RuntimeError(f"No tuning candidate evaluated for {method}")
    chosen = dict(best[1])
    tuning = pd.DataFrame(rows)
    tuning["selected"] = False
    mask = (
        (tuning["method_code"] == chosen["method_code"])
        & (tuning["eta"] == chosen["eta"])
        & (tuning["alpha"] == chosen["alpha"])
        & (tuning["beta"] == chosen["beta"])
        & (tuning["rho"] == chosen["rho"])
    )
    tuning.loc[mask, "selected"] = True
    chosen["selected"] = True
    return chosen, best[2], tuning


def refined_grid(center: float, radius: float, step: float) -> list[float]:
    lower = max(-10.0, center - radius)
    upper = min(10.0, center + radius)
    return [float(value) for value in np.arange(lower, upper + 0.5 * step, step)]


def tune_particle_spec_full(
    train: list[base.Record],
    val: list[base.Record],
    config: dict[str, Any],
    use_diversity: bool,
    horizon_days: int,
    objective_baseline: dict[str, float],
) -> tuple[benchmarks.ParticleSpec, pd.DataFrame]:
    pf = config["benchmarks"]["particle_filter"]
    label = "dtvw" if use_diversity else "tvw"
    rows: list[dict[str, Any]] = []

    def evaluate_grid(alpha1_grid: list[float], alpha2_grid: list[float], stage: str) -> tuple[float, benchmarks.ParticleSpec]:
        best: tuple[float, benchmarks.ParticleSpec] | None = None
        total = len(alpha1_grid) * len(alpha2_grid)
        count = 0
        for alpha1 in alpha1_grid:
            for alpha2 in alpha2_grid:
                count += 1
                spec = benchmarks.particle_spec(config, float(alpha1), float(alpha2))
                metrics = tuning_panel.particle(spec, use_diversity, horizon_days)
                objective = validation_objective_from_base(metrics, objective_baseline)
                rows.append(
                    {
                        "method_type": "particle_filter",
                        "method_code": label,
                        "search_stage": stage,
                        "alpha0": spec.alpha0,
                        "alpha1": spec.alpha1,
                        "alpha2": spec.alpha2,
                        "particles": spec.particles,
                        "state_noise": spec.state_noise,
                        "coefficient_noise": spec.coefficient_noise,
                        "validation_objective": objective,
                        "validation_CRPS": float(metrics["crps"]),
                        "validation_twCRPS_U": float(metrics["twcrps_u"]),
                        "validation_Brier_U": float(metrics["brier_u"]),
                        "validation_QS95": validation_metric_value(metrics, "qs95"),
                        "validation_ECAL": float(metrics["ecal"]),
                        "validation_Cost": float(metrics["newsvendor_cost"]),
                        "validation_Service": float(metrics["service_level"]),
                    }
                )
                if best is None or objective < best[0]:
                    best = (objective, spec)
        print(f"Tuned {label} {stage}: {count}/{total} candidate(s)", flush=True)
        if best is None:
            raise RuntimeError(f"No {label} particle candidate evaluated in {stage}")
        return best

    tuning_panel = TuningPanel(train, val)
    coarse_alpha1 = [float(value) for value in pf["alpha1_grid"]]
    coarse_alpha2 = [float(value) for value in pf["alpha2_grid"]] if use_diversity else [0.0]
    coarse_best = evaluate_grid(coarse_alpha1, coarse_alpha2, "coarse")
    radius = float(pf["refine_radius"])
    step = float(pf["refine_step"])
    fine_alpha1 = refined_grid(coarse_best[1].alpha1, radius, step)
    fine_alpha2 = refined_grid(coarse_best[1].alpha2, radius, step) if use_diversity else [0.0]
    fine_best = evaluate_grid(fine_alpha1, fine_alpha2, "refined")

    tuning = pd.DataFrame(rows)
    tuning["selected"] = False
    tuning.loc[
        (tuning["alpha1"] == fine_best[1].alpha1)
        & (tuning["alpha2"] == fine_best[1].alpha2)
        & (tuning["search_stage"] == "refined"),
        "selected",
    ] = True
    return fine_best[1], tuning


def append_weight_rows(rows: list[dict[str, Any]], method: str, split: str, origin: int, region_weights: dict[str, np.ndarray]) -> None:
    for region, weights in region_weights.items():
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 0.0)
        weights = weights / max(float(weights.sum()), 1e-12)
        for model, weight in zip(POOL_MODELS, weights):
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "origin": int(origin),
                    "region": region,
                    "model_name": model,
                    "weight": float(weight),
                }
            )


def evaluate_constant_pool(
    records: list[base.Record],
    method: str,
    split: str,
    weights: np.ndarray | None,
    forecast: Callable[[base.Record], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    upper_weights: list[np.ndarray] = []
    weight_rows: list[dict[str, Any]] = []
    last_origin: int | None = None
    if weights is None:
        weights = np.repeat(1.0 / len(POOL_MODELS), len(POOL_MODELS))
    for rec in sorted(records, key=lambda r: (r.day, r.series_id)):
        if forecast is benchmarks.quantile_average_pool:
            row = quantile_average.prediction_row(rec)
            row["origin"] = int(row.pop("day"))
            row["actual_D28"] = float(row["y"])
            row["q95_bias_point"] = float(row["q95"] - row["y"])
            row["q95_coverage_point"] = float(row["y"] <= row["q95"])
        elif forecast is None:
            pdf, cdf = benchmarks.linear_pool(rec, weights)
            row = prediction_row_with_quantiles(rec, pdf, cdf)
        else:
            pdf, cdf = forecast(rec)
            row = prediction_row_with_quantiles(rec, pdf, cdf)
        detail_rows.append(row)
        upper_weights.append(np.asarray(weights, dtype=float).copy())
        if last_origin != rec.day:
            append_weight_rows(weight_rows, method, split, int(rec.day), {"global": weights})
            last_origin = int(rec.day)
    detail = pd.DataFrame(detail_rows)
    metrics = summarize_detail(detail, records, method, split, upper_weights)
    return metrics, detail, pd.DataFrame(weight_rows)


def evaluate_core_delayed_trace(
    score_records: list[base.Record],
    method: str,
    label: str,
    split: str,
    static_weights: dict[str, np.ndarray],
    alpha: float,
    beta: float,
    rho: float,
    horizon_days: int,
    warm_records: list[base.Record] | None = None,
    enqueue_score_updates: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    k = len(POOL_MODELS)
    theta_g, theta_c, theta_l, theta_u, omega_g, omega_c, omega_l, omega_u = base.initialize_score_driven_states(
        static_weights, beta
    )
    pending: list[tuple[int, dict[str, Any]]] = []
    detail_rows: list[dict[str, Any]] = []
    upper_weights: list[np.ndarray] = []
    weight_rows: list[dict[str, Any]] = []
    update_events: list[dict[str, int]] = []

    def apply_available_updates(current_day: int) -> None:
        nonlocal pending, theta_g, theta_c, theta_l, theta_u
        available, blocked = timing.split_available_updates(pending, current_day, horizon_days)
        pending = blocked
        for update_origin, payload in available:
            update_events.append({"update_origin": int(update_origin), "current_origin": int(current_day)})
            day_records = payload["records"]
            if method == "dynamic_global":
                update_w = payload["w_global"]
                scores = [base.auxiliary_logit_score(rec, update_w, "global") for rec in day_records]
                theta_g = base.update_dynamic_global_state(
                    theta_g, omega_g, alpha, beta, np.mean(scores, axis=0)
                )
            elif method == "sd_ralp":
                scores_c = [base.auxiliary_logit_score(rec, payload["w_c"], "C") for rec in day_records]
                scores_l = [base.auxiliary_logit_score(rec, payload["w_l"], "L") for rec in day_records]
                scores_u = [base.auxiliary_logit_score(rec, payload["w_u"], "U") for rec in day_records]
                theta_c, theta_l, theta_u = base.update_sd_ralp_states(
                    theta_c,
                    theta_l,
                    theta_u,
                    omega_c,
                    omega_l,
                    omega_u,
                    alpha,
                    beta,
                    rho,
                    np.mean(scores_c, axis=0),
                    np.mean(scores_l, axis=0),
                    np.mean(scores_u, axis=0),
                )

    def process(records: list[base.Record], collect: bool, enqueue_updates: bool) -> None:
        ordered = sorted(records, key=lambda rec: (rec.day, rec.series_id))
        for day, day_group in groupby(ordered, key=lambda rec: rec.day):
            day_records = list(day_group)
            apply_available_updates(int(day))
            if method == "equal":
                w_l = w_c = w_u = np.repeat(1.0 / k, k)
                region_weights = {"global": w_u}
            elif method == "dynamic_global":
                w = base.softmax(theta_g)
                w_l = w_c = w_u = w
                region_weights = {"global": w_u}
            elif method == "static_ralp":
                w_l, w_c, w_u = static_weights["L"], static_weights["C"], static_weights["U"]
                region_weights = {"L": w_l, "C": w_c, "U": w_u}
            elif method == "sd_ralp":
                w_l, w_c, w_u = base.softmax(theta_l), base.softmax(theta_c), base.softmax(theta_u)
                region_weights = {"L": w_l, "C": w_c, "U": w_u}
            else:
                raise ValueError(method)
            if collect:
                append_weight_rows(weight_rows, label, split, int(day), region_weights)
                for rec in day_records:
                    pdf, cdf = base.combine_record(rec, w_l, w_c, w_u)
                    detail_rows.append(prediction_row_with_quantiles(rec, pdf, cdf))
                    upper_weights.append(np.asarray(w_u, dtype=float).copy())
            if enqueue_updates and method == "dynamic_global":
                pending.append((int(day), {"records": day_records, "w_global": np.asarray(w_l, dtype=float).copy()}))
            elif enqueue_updates and method == "sd_ralp":
                pending.append((int(day), {"records": day_records, "w_l": np.asarray(w_l, dtype=float).copy(), "w_c": np.asarray(w_c, dtype=float).copy(), "w_u": np.asarray(w_u, dtype=float).copy()}))

    process(warm_records or [], collect=False, enqueue_updates=True)
    process(score_records, collect=True, enqueue_updates=enqueue_score_updates)
    detail = pd.DataFrame(detail_rows)
    metrics = summarize_detail(detail, score_records, label, split, upper_weights)
    return metrics, detail, pd.DataFrame(weight_rows), update_events


def evaluate_recursive_gdfc_trace(
    history_records: list[base.Record],
    score_records: list[base.Record],
    split: str,
    horizon_days: int,
    enqueue_score_updates: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    """Recursive GDFC (Kapetanios et al., 2015), refitted at every origin on all completed outcomes."""
    pending: list[tuple[int, list[base.Record]]] = benchmarks.day_groups(history_records)
    realised_history: list[base.Record] = []
    realised_origins: list[int] = []
    logits: np.ndarray | None = None
    detail_rows: list[dict[str, Any]] = []
    upper_weights: list[np.ndarray] = []
    weight_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    update_events: list[dict[str, int]] = []

    for day, group in benchmarks.day_groups(score_records):
        available, pending = timing.split_available_updates(pending, day, horizon_days)
        for update_origin, update_records in available:
            if update_origin + horizon_days - 1 >= day:
                raise RuntimeError("Recursive GDFC attempted to use an unavailable outcome")
            realised_history.extend(update_records)
            realised_origins.append(int(update_origin))
            update_events.append({"update_origin": int(update_origin), "current_origin": int(day)})
        if not realised_history:
            raise RuntimeError(f"Recursive GDFC has no realised history at origin {day}")

        fit = benchmarks.fit_gdfc_coefficients(realised_history, initial_logits=logits)
        logits = fit.logits.copy()
        coefficients = fit.coefficients
        coefficient_sum = float(coefficients.sum())
        if not np.isfinite(coefficients).all() or np.any(coefficients <= 0.0) or abs(coefficient_sum - 1.0) > 1e-8:
            raise RuntimeError(f"Invalid recursive GDFC coefficients at origin {day}")
        fit_rows.append(
            {
                "method_type": "recursive_gdfc",
                "method_code": "recursive_gdfc_kmpf2015",
                "method": GDFC_LABEL,
                "split": split,
                "origin": int(day),
                "n_history_origins": int(len(realised_origins)),
                "n_history_records": int(fit.n_records),
                "objective_log_score": float(fit.objective),
                "optimizer_success": bool(fit.success),
                "optimizer_message": fit.message,
                "optimizer_iterations": int(fit.iterations),
                "gradient_norm": float(fit.gradient_norm),
                "coefficient_sum": coefficient_sum,
                "selected": True,
                "note": "three pre-specified operational regions; recursively estimated using only fully realised lead-time outcomes",
            }
        )

        for region_idx, region in enumerate(benchmarks.GDFC_REGIONS):
            raw = coefficients[:, region_idx]
            region_scale = float(raw.sum())
            conditional = raw / max(region_scale, 1e-12)
            for model, weight, raw_coefficient in zip(POOL_MODELS, conditional, raw):
                weight_rows.append(
                    {
                        "method": GDFC_LABEL,
                        "split": split,
                        "origin": int(day),
                        "region": region,
                        "model_name": model,
                        "weight": float(weight),
                        "raw_coefficient": float(raw_coefficient),
                        "region_scale": region_scale,
                    }
                )

        upper_share = coefficients[:, 2] / max(float(coefficients[:, 2].sum()), 1e-12)
        for rec in group:
            pdf, cdf = benchmarks.gdfc_pool(rec, coefficients)
            detail_rows.append(prediction_row_with_quantiles(rec, pdf, cdf))
            upper_weights.append(upper_share.copy())
        if enqueue_score_updates:
            pending.append((int(day), group))

    detail = pd.DataFrame(detail_rows)
    metrics = summarize_detail(detail, score_records, GDFC_LABEL, split, upper_weights)
    return metrics, detail, pd.DataFrame(weight_rows), pd.DataFrame(fit_rows), update_events


def evaluate_particle_trace(
    train_records: list[base.Record],
    warm_records: list[base.Record],
    score_records: list[base.Record],
    split: str,
    spec: benchmarks.ParticleSpec,
    use_diversity: bool,
    horizon_days: int,
    enqueue_score_updates: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    label = "DTVW" if use_diversity else "TVW"
    state = benchmarks.initial_particle_state(spec, len(POOL_MODELS))
    pending: list[tuple[int, dict[str, np.ndarray]]] = []
    detail_rows: list[dict[str, Any]] = []
    upper_weights: list[np.ndarray] = []
    weight_rows: list[dict[str, Any]] = []
    update_events: list[dict[str, int]] = []

    def apply_available(current_day: int) -> None:
        nonlocal pending
        available, pending = timing.split_available_updates(pending, current_day, horizon_days)
        idx = 0
        while idx < len(available):
            update_origin, payload = available[idx]
            update_events.append({"update_origin": int(update_origin), "current_origin": int(current_day)})
            benchmarks.apply_delayed_particle_likelihood(state, spec, payload)
            pending, remaining = benchmarks.maybe_resample_particles(state, spec, pending, available[idx + 1 :])
            available = available[: idx + 1] + remaining
            idx += 1

    def process(records: list[base.Record], collect: bool, enqueue_updates: bool) -> None:
        for day, group in benchmarks.day_groups(records):
            apply_available(day)
            state.alpha += state.rng.normal(0.0, spec.coefficient_noise, size=state.alpha.shape)
            theta = 2.0 * (1.0 / (1.0 + np.exp(-state.alpha)) - 0.5)
            diversity = benchmarks.panel_diversity(group) if use_diversity else np.zeros(state.x.shape[1])
            state.x = (
                theta[:, 0, None]
                + theta[:, 1, None] * state.x
                + theta[:, 2, None] * diversity[None, :]
                + state.rng.normal(0.0, spec.state_noise, size=state.x.shape)
            )
            state.x = np.clip(state.x, -12.0, 12.0)
            component_weights = benchmarks.row_softmax(state.x)
            particle_weights = benchmarks.stable_softmax(state.log_weights)
            effective_weights = particle_weights @ component_weights
            if collect:
                append_weight_rows(weight_rows, label, split, int(day), {"global": effective_weights})
                for rec in group:
                    pdf, cdf = benchmarks.linear_pool(rec, effective_weights)
                    detail_rows.append(prediction_row_with_quantiles(rec, pdf, cdf))
                    upper_weights.append(effective_weights.copy())
            if enqueue_updates:
                pending.append((int(day), {"density": np.vstack([benchmarks.component_density_at_y(rec) for rec in group]), "component_weights": component_weights.copy()}))

    process(train_records + warm_records, collect=False, enqueue_updates=True)
    process(score_records, collect=True, enqueue_updates=enqueue_score_updates)
    detail = pd.DataFrame(detail_rows)
    metrics = summarize_detail(detail, score_records, label, split, upper_weights)
    return metrics, detail, pd.DataFrame(weight_rows), update_events


def output_predictions(details: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(details, ignore_index=True)
    cols_first = ["method", "split", "origin", "series_id", "actual_D28", *QCOLS]
    metric_cols = [
        "crps",
        "twcrps_u",
        "log_score",
        "tail_log_score",
        "newsvendor_cost",
        "stockout",
        "upper_event",
        "upper_event_prob",
        "brier_u",
        "pit",
        "q_tau",
        "cvar_tau",
    ]
    rest = [c for c in out.columns if c not in cols_first + metric_cols + ["y"]]
    return out[cols_first + metric_cols + rest]
