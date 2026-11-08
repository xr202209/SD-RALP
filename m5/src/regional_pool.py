"""Regional linear pool for the M5 application.

Records, scores and the newsvendor cost, static weights, and the score-driven
updates of Global GAS and SD-RALP (with centre-tail shrinkage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


MODEL_NAMES = ["LightGBMQuantile", "DeepAR", "KNNAnalogue", "GAMEVT", "LocalGAMQuantile"]


@dataclass
class Record:
    series_id: str
    cat_id: str
    day: int
    y: float
    tau: float
    grid: np.ndarray
    comp_pdfs: np.ndarray
    comp_cdfs: np.ndarray
    c_low: float
    c_high: float
    obs_region: str
    model_global: np.ndarray
    model_l: np.ndarray
    model_c: np.ndarray
    model_u: np.ndarray


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    ez = np.exp(z)
    return ez / ez.sum()


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    return float(np.trapezoid(y, x))


def cumulative_from_pdf(grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    dz = float(grid[1] - grid[0])
    cdf = np.cumsum(np.maximum(pdf, 0.0)) * dz
    cdf = cdf / max(float(cdf[-1]), 1e-12)
    cdf[-1] = 1.0
    return cdf


def quantile_from_cdf(grid: np.ndarray, cdf: np.ndarray, tau: float) -> float:
    idx = int(np.searchsorted(cdf, tau, side="left"))
    if idx <= 0:
        return float(grid[0])
    if idx >= len(grid):
        return float(grid[-1])
    x0, x1 = grid[idx - 1], grid[idx]
    y0, y1 = cdf[idx - 1], cdf[idx]
    if y1 <= y0:
        return float(x1)
    return float(x0 + (tau - y0) * (x1 - x0) / (y1 - y0))


def score_losses(grid: np.ndarray, cdf: np.ndarray, y: float, c_low: float, c_high: float) -> tuple[float, float, float, float]:
    dz = float(grid[1] - grid[0])
    indicator = (y <= grid).astype(float)
    err = (cdf - indicator) ** 2
    mask_l = grid <= c_low
    mask_c = (grid > c_low) & (grid <= c_high)
    mask_u = grid > c_high
    # Above the finite forecast support F=1. If y exceeds it, the squared
    # CDF error is exactly one up to y; include that interval analytically.
    excess = max(float(y) - float(grid[-1]), 0.0)
    excess_l = max(min(float(y), c_low) - float(grid[-1]), 0.0)
    excess_c = max(min(float(y), c_high) - max(float(grid[-1]), c_low), 0.0)
    excess_u = max(float(y) - max(float(grid[-1]), c_high), 0.0)
    return (
        float(err.sum() * dz + excess),
        float(err[mask_l].sum() * dz + excess_l),
        float(err[mask_c].sum() * dz + excess_c),
        float(err[mask_u].sum() * dz + excess_u),
    )


def newsvendor_cost(q: float, y: float, tau: float) -> float:
    c_over = 1.0
    c_under = tau / max(1.0 - tau, 1e-6)
    return c_over * max(q - y, 0.0) + c_under * max(y - q, 0.0)


def density_at_value(grid: np.ndarray, pdf: np.ndarray, value: float) -> float:
    return float(max(np.interp(value, grid, pdf, left=0.0, right=0.0), 1e-12))


def cdf_at_value(grid: np.ndarray, cdf: np.ndarray, value: float) -> float:
    return float(np.clip(np.interp(value, grid, cdf, left=0.0, right=1.0), 0.0, 1.0))


def cvar_from_pdf(grid: np.ndarray, pdf: np.ndarray, cdf: np.ndarray, tau: float) -> float:
    q_tau = quantile_from_cdf(grid, cdf, tau)
    mask = grid >= q_tau
    if int(mask.sum()) < 2:
        return float(q_tau)
    tail_mass = trapz(pdf[mask], grid[mask])
    if tail_mass <= 1e-12:
        return float(q_tau)
    return float(trapz(grid[mask] * pdf[mask], grid[mask]) / tail_mass)


def prediction_row(rec: Record, pdf: np.ndarray, cdf: np.ndarray) -> dict[str, float | str]:
    global_loss, _, _, upper_loss = score_losses(rec.grid, cdf, rec.y, rec.c_low, rec.c_high)
    q_tau = quantile_from_cdf(rec.grid, cdf, rec.tau)
    density_y = density_at_value(rec.grid, pdf, rec.y)
    cdf_high = max(cdf_at_value(rec.grid, cdf, rec.c_high), 1e-12)
    upper_event = float(rec.y > rec.c_high)
    upper_prob = float(np.clip(1.0 - cdf_high, 0.0, 1.0))
    tail_log_score = -math.log(density_y) if rec.y > rec.c_high else -math.log(cdf_high)
    return {
        "day": rec.day,
        "series_id": rec.series_id,
        "y": rec.y,
        "crps": global_loss,
        "twcrps_u": upper_loss,
        "log_score": -math.log(density_y),
        "tail_log_score": tail_log_score,
        "pit": cdf_at_value(rec.grid, cdf, rec.y),
        "newsvendor_cost": newsvendor_cost(q_tau, rec.y, rec.tau),
        "stockout": float(rec.y > q_tau),
        "upper_event": upper_event,
        "upper_event_prob": upper_prob,
        "brier_u": float((upper_event - upper_prob) ** 2),
        "q_tau": q_tau,
        "cvar_tau": cvar_from_pdf(rec.grid, pdf, cdf, rec.tau),
    }


def summarize_predictions(
    rows: pd.DataFrame,
    records: list[Record],
    upper_weights: list[np.ndarray] | None = None,
    model_names: list[str] | None = None,
) -> dict[str, float]:
    target_stockout = float(np.mean([1.0 - rec.tau for rec in records]))
    series_cvar_errors = []
    series_empirical_cvars = []
    for _, scoped in rows.groupby("series_id"):
        empirical_threshold = float(np.quantile(scoped["y"], 1.0 - target_stockout))
        empirical_tail = scoped.loc[scoped["y"] >= empirical_threshold, "y"]
        empirical_cvar = float(empirical_tail.mean()) if not empirical_tail.empty else empirical_threshold
        series_empirical_cvars.append(empirical_cvar)
        series_cvar_errors.append(abs(float(scoped["cvar_tau"].mean()) - empirical_cvar))
    result = {
        "crps": float(rows["crps"].mean()),
        "twcrps_u": float(rows["twcrps_u"].mean()),
        "log_score": float(rows["log_score"].mean()),
        "tail_log_score": float(rows["tail_log_score"].mean()),
        "newsvendor_cost": float(rows["newsvendor_cost"].mean()),
        "stockout_rate": float(rows["stockout"].mean()),
        "service_level": float(1.0 - rows["stockout"].mean()),
        "ecal": float(abs(rows["stockout"].mean() - target_stockout)),
        "brier_u": float(rows["brier_u"].mean()) if "brier_u" in rows else float("nan"),
        "mean_q_tau": float(rows["q_tau"].mean()),
        "mean_pit": float(rows["pit"].mean()),
        "pit_variance": float(rows["pit"].var(ddof=0)),
        "mean_cvar_tau": float(rows["cvar_tau"].mean()),
        "empirical_cvar_tau": float(np.mean(series_empirical_cvars)),
        "cvar_abs_error": float(np.mean(series_cvar_errors)),
    }
    if upper_weights:
        upper_mean = np.mean(np.vstack(upper_weights), axis=0)
        for idx, name in enumerate(model_names or MODEL_NAMES):
            result[f"upper_w_{name}_mean"] = float(upper_mean[idx])
    return result


def combine_record(rec: Record, w_l: np.ndarray, w_c: np.ndarray, w_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_l = rec.grid <= rec.c_low
    mask_c = (rec.grid > rec.c_low) & (rec.grid <= rec.c_high)
    mask_u = rec.grid > rec.c_high
    pdf = np.zeros_like(rec.grid)
    pdf[mask_l] = (w_l[:, None] * rec.comp_pdfs[:, mask_l]).sum(axis=0)
    pdf[mask_c] = (w_c[:, None] * rec.comp_pdfs[:, mask_c]).sum(axis=0)
    pdf[mask_u] = (w_u[:, None] * rec.comp_pdfs[:, mask_u]).sum(axis=0)
    pdf = pdf / max(trapz(pdf, rec.grid), 1e-12)
    return pdf, cumulative_from_pdf(rec.grid, pdf)


def regional_mask(rec: Record, region: str) -> np.ndarray:
    if region == "global":
        return np.ones_like(rec.grid, dtype=bool)
    if region == "L":
        return rec.grid <= rec.c_low
    if region == "C":
        return (rec.grid > rec.c_low) & (rec.grid <= rec.c_high)
    if region == "U":
        return rec.grid > rec.c_high
    raise ValueError(f"Unknown region: {region}")


def regional_model_losses(rec: Record, region: str) -> np.ndarray:
    if region == "global":
        return rec.model_global
    if region == "L":
        return rec.model_l
    if region == "C":
        return rec.model_c
    if region == "U":
        return rec.model_u
    raise ValueError(f"Unknown region: {region}")


def regional_disagreement(rec: Record, region: str) -> np.ndarray:
    cache = getattr(rec, "_disagreement_cache", None)
    if cache is None:
        cache = {}
        setattr(rec, "_disagreement_cache", cache)
    if region in cache:
        return cache[region]
    mask = regional_mask(rec, region)
    cdfs = rec.comp_cdfs[:, mask]
    dz = float(rec.grid[1] - rec.grid[0])
    diff = cdfs[:, None, :] - cdfs[None, :, :]
    cache[region] = np.sum(diff * diff, axis=2) * dz
    return cache[region]


def auxiliary_logit_score(rec: Record, weights: np.ndarray, region: str) -> np.ndarray:
    b = regional_model_losses(rec, region)
    disagreement = regional_disagreement(rec, region)
    direct_gradient = b - disagreement @ weights
    return weights * (float(np.dot(weights, direct_gradient)) - direct_gradient)


def fit_auxiliary_weights(records: list[Record], region: str, eta: float) -> np.ndarray:
    mean_losses = np.mean([regional_model_losses(rec, region) for rec in records], axis=0)
    mean_disagreement = np.mean([regional_disagreement(rec, region) for rec in records], axis=0)
    theta = -eta * mean_losses
    for _ in range(250):
        weights = softmax(theta)
        direct_gradient = mean_losses - mean_disagreement @ weights
        score = weights * (float(np.dot(weights, direct_gradient)) - direct_gradient)
        theta = np.clip(theta + 0.35 * score, -12.0, 12.0)
    return softmax(theta)


def fit_static_weights(records: list[Record], eta: float) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("Static weights require at least one forecast record")
    return {
        "global": fit_auxiliary_weights(records, "global", eta),
        "L": fit_auxiliary_weights(records, "L", eta),
        "C": fit_auxiliary_weights(records, "C", eta),
        "U": fit_auxiliary_weights(records, "U", eta),
    }


def initialize_score_driven_states(
    static_weights: dict[str, np.ndarray],
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    theta_g = np.log(static_weights["global"] + eps)
    theta_c = np.log(static_weights["C"] + eps)
    theta_l = np.log(static_weights["L"] + eps)
    theta_u = np.log(static_weights["U"] + eps)
    omega_g = (1.0 - beta) * theta_g
    omega_c = (1.0 - beta) * theta_c
    omega_l = (1.0 - beta) * theta_l
    omega_u = (1.0 - beta) * theta_u
    return theta_g, theta_c, theta_l, theta_u, omega_g, omega_c, omega_l, omega_u


def update_dynamic_global_state(
    theta_g: np.ndarray,
    omega_g: np.ndarray,
    alpha: float,
    beta: float,
    mean_scores: np.ndarray,
) -> np.ndarray:
    return np.clip(omega_g + beta * theta_g + alpha * mean_scores, -7.0, 7.0)


def update_sd_ralp_states(
    theta_c: np.ndarray,
    theta_l: np.ndarray,
    theta_u: np.ndarray,
    omega_c: np.ndarray,
    omega_l: np.ndarray,
    omega_u: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    mean_scores_c: np.ndarray,
    mean_scores_l: np.ndarray,
    mean_scores_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_c_new = np.clip(omega_c + beta * theta_c + alpha * mean_scores_c, -7.0, 7.0)
    theta_l_free = np.clip(omega_l + beta * theta_l + alpha * mean_scores_l, -7.0, 7.0)
    theta_u_free = np.clip(omega_u + beta * theta_u + alpha * mean_scores_u, -7.0, 7.0)
    theta_l_new = np.clip((1.0 - rho) * theta_l_free + rho * theta_c_new, -7.0, 7.0)
    theta_u_new = np.clip((1.0 - rho) * theta_u_free + rho * theta_c_new, -7.0, 7.0)
    return theta_c_new, theta_l_new, theta_u_new
