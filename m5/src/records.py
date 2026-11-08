"""Forecast records of the M5 application.

Each record is one replenishment decision: a series, a weekly forecast origin
and the demand over the next 28 days, together with the five component
predictive distributions on a common grid.  Component forecasts are given by
their 0.10, 0.50, 0.90, 0.95 and 0.99 quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import regional_pool as base

HORIZON = 28
ORIGIN_START = 1004
ORIGIN_END = 1914
ORIGIN_STEP = 7
TRAIN_ORIGIN_COUNT = 30
VALIDATION_ORIGIN_COUNT = 40
QUANTILE_LEVELS = [0.10, 0.50, 0.90, 0.95, 0.99]
QCOLS = ["q10", "q50", "q90", "q95", "q99"]
MODEL_NAMES = base.MODEL_NAMES


@dataclass(frozen=True)
class LeadTimeForecasts:
    origins: list[int]
    horizon_days: int
    model_quantiles: dict[str, dict[float, dict[tuple[str, int], float]]]


def origins() -> list[int]:
    return list(range(ORIGIN_START, ORIGIN_END + 1, ORIGIN_STEP))


def day_cols(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith("d_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def quantile_maps(frames: dict[str, pd.DataFrame]) -> dict[str, dict[float, dict[tuple[str, int], float]]]:
    out: dict[str, dict[float, dict[tuple[str, int], float]]] = {}
    for model, frame in frames.items():
        model_map: dict[float, dict[tuple[str, int], float]] = {q: {} for q in QUANTILE_LEVELS}
        keys = list(zip(frame["series_id"].astype(str), frame["origin"].astype(int)))
        for q, col in zip(QUANTILE_LEVELS, QCOLS):
            model_map[float(q)] = {key: float(value) for key, value in zip(keys, frame[col].to_numpy(dtype=float))}
        out[model] = model_map
    return out


def component_quantile_values(
    quantiles: dict[float, dict[tuple[str, int], float]],
    sid: str,
    day: int,
) -> np.ndarray:
    return np.maximum.accumulate([quantiles[a][(sid, day)] for a in QUANTILE_LEVELS])


def five_quantile_pdf_cdf(
    grid: np.ndarray,
    predicted: np.ndarray,
    support_upper: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear CDF through the five quantiles and two support anchors."""

    q10, q50, q90, q95, q99 = [float(x) for x in predicted]
    lower_span = max(q50 - q10, 0.75)
    upper_span = max(q99 - q95, q95 - q90, 0.75)
    anchor_probs = np.array([0.0, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0])
    terminal_upper = max(
        q99 + 2.0 * upper_span,
        float(grid[-1]) if support_upper is None else float(support_upper),
    )
    anchor_values = np.array(
        [
            max(0.0, q10 - 1.5 * lower_span),
            q10,
            q50,
            q90,
            q95,
            q99,
            terminal_upper,
        ]
    )
    anchor_values = np.maximum.accumulate(anchor_values)
    # Tiny increments keep interpolation well-defined when quantile models tie.
    anchor_values = anchor_values + np.arange(len(anchor_values)) * 1e-6
    cdf = np.interp(grid, anchor_values, anchor_probs, left=0.0, right=1.0)
    pdf = np.gradient(cdf, grid)
    pdf = np.maximum(pdf, 0.0)
    pdf = pdf / max(base.trapz(pdf, grid), 1e-12)
    return pdf, base.cumulative_from_pdf(grid, pdf)


def build_records(
    sales: pd.DataFrame,
    forecasts: LeadTimeForecasts,
    tau: float,
    grid_size: int,
) -> list[base.Record]:
    cols = day_cols(sales)
    records = []
    for _, row in sales.iterrows():
        sid = str(row["id"])
        values = row[cols].to_numpy(dtype=float)
        for origin in forecasts.origins:
            y = float(values[origin - 1 : origin - 1 + forecasts.horizon_days].sum())
            model_q = [
                component_quantile_values(forecasts.model_quantiles[name], sid, origin)
                for name in MODEL_NAMES
            ]
            support_upper = [
                float(quantiles[-1] + 2.0 * max(quantiles[-1] - quantiles[-2], 1.0))
                for quantiles in model_q
            ]
            # The grid depends on the forecasts only. Demand beyond it is
            # scored analytically.
            grid_upper = max(max(support_upper), 10.0)
            grid = np.linspace(0.0, grid_upper, grid_size)
            components = [
                five_quantile_pdf_cdf(grid, quantiles, model_support)
                for quantiles, model_support in zip(model_q, support_upper)
            ]
            comp_pdfs = np.vstack([pdf for pdf, _ in components])
            comp_cdfs = np.vstack([cdf for _, cdf in components])
            # Regional thresholds: the 0.10 and tau - 0.05 quantiles of the equal-weight pool.
            equal_pdf = comp_pdfs.mean(axis=0)
            equal_pdf /= max(base.trapz(equal_pdf, grid), 1e-12)
            equal_cdf = base.cumulative_from_pdf(grid, equal_pdf)
            tau0 = max(0.05, min(0.95, tau - 0.05))
            c_low = base.quantile_from_cdf(grid, equal_cdf, 0.10)
            c_high = base.quantile_from_cdf(grid, equal_cdf, tau0)
            obs_region = "L" if y <= c_low else "C" if y <= c_high else "U"
            losses = np.array([base.score_losses(grid, cdf, y, c_low, c_high) for cdf in comp_cdfs])
            records.append(
                base.Record(
                    series_id=sid,
                    cat_id=str(row["cat_id"]),
                    day=origin,
                    y=y,
                    tau=tau,
                    grid=grid,
                    comp_pdfs=comp_pdfs,
                    comp_cdfs=comp_cdfs,
                    c_low=c_low,
                    c_high=c_high,
                    obs_region=obs_region,
                    model_global=losses[:, 0],
                    model_l=losses[:, 1],
                    model_c=losses[:, 2],
                    model_u=losses[:, 3],
                )
            )
    return records


def split_records(
    records: list[base.Record], all_origins: list[int]
) -> tuple[list[base.Record], list[base.Record], list[base.Record]]:
    train_origins = set(all_origins[:TRAIN_ORIGIN_COUNT])
    val_origins = set(all_origins[TRAIN_ORIGIN_COUNT : TRAIN_ORIGIN_COUNT + VALIDATION_ORIGIN_COUNT])
    test_origins = set(all_origins[TRAIN_ORIGIN_COUNT + VALIDATION_ORIGIN_COUNT :])
    return (
        [rec for rec in records if rec.day in train_origins],
        [rec for rec in records if rec.day in val_origins],
        [rec for rec in records if rec.day in test_origins],
    )
