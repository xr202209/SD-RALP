"""Price data, component forecasts and forecast scores for the electricity application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from combination import CombinationResult, PanelData, ParticleSpec, RegionSpec
from distributions import (
    discrete_crps,
    discrete_log_score,
    discrete_lower_twcrps,
    discrete_upper_twcrps,
    make_grid,
    normalize_pmf,
    quantiles_to_discrete,
)

SLOTS_PER_DAY = 96
FINAL_QUANTILES = np.round(np.arange(0.01, 1.00, 0.01), 2)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_price_curves(csv_path: Path, target: str, last_day: str) -> pd.DataFrame:
    """Return realised quarter-hourly prices as one row of 96 slots per delivery day.

    Timestamps label interval ends, so ``00:00`` belongs to the previous delivery day.
    Rows after ``last_day`` are not used.
    """

    raw = pd.read_csv(csv_path, usecols=["datetime", target], parse_dates=["datetime"])
    start = raw["datetime"] - pd.Timedelta(minutes=15)
    raw["delivery_day"] = start.dt.normalize()
    raw["slot"] = start.dt.hour * 4 + start.dt.minute // 15 + 1
    raw = raw.loc[raw["delivery_day"] <= pd.Timestamp(last_day)]
    days = pd.date_range(raw["delivery_day"].min(), pd.Timestamp(last_day), freq="D")
    curves = raw.pivot(index="delivery_day", columns="slot", values=target)
    curves = curves.reindex(index=days, columns=np.arange(1, SLOTS_PER_DAY + 1))
    if curves.isna().any().any():
        raise RuntimeError("price file has missing quarter-hours before the last test day")
    curves.index.name = "delivery_day"
    return curves.astype(float)


def forecast_days(split: Mapping[str, str]) -> pd.DatetimeIndex:
    """Combination-training, validation and test delivery days in time order."""

    return pd.date_range(split["train_start"], split["test_end"], freq="D")


def load_component_panel(
    config: Mapping[str, Any],
    component_dir: Path,
    curves: pd.DataFrame,
) -> PanelData:
    """Load the component forecasts for all forecast days on the common price grid."""

    days = forecast_days(config["split"])
    models = list(config["components"])
    bridge = config["distribution_bridge"]
    support = (float(bridge["support_lower"]), float(bridge["support_upper"]))
    grid = make_grid(support=support, step=float(bridge["grid_step"]))
    pmf = np.empty((len(days), SLOTS_PER_DAY, len(models), len(grid)), dtype=np.float32)
    cdf = np.empty_like(pmf)
    expected_days = days.to_numpy(dtype="datetime64[D]")
    for model_index, model in enumerate(models):
        path = Path(component_dir) / f"{model}.npz"
        with np.load(path, allow_pickle=False) as payload:
            delivery_days = payload["delivery_days"].astype("datetime64[D]")
            levels = payload["quantile_levels"].astype(float)
            quantiles = payload["quantiles"].astype(float)
            p_zero = payload["p_zero"].astype(float)
            fit_end = payload["fit_end_by_day"].astype("datetime64[D]")
            direct_pmf = payload["pmf"].astype(float) if "pmf" in payload.files else None
            direct_grid = payload["pmf_grid"].astype(float) if "pmf" in payload.files else None
        if not np.array_equal(delivery_days, expected_days):
            raise RuntimeError(f"{model}: delivery days do not match the sample split")
        if not np.allclose(levels, FINAL_QUANTILES, atol=1e-7, rtol=0.0):
            raise RuntimeError(f"{model}: quantile levels must be 0.01, ..., 0.99")
        if np.any(fit_end >= delivery_days):
            raise RuntimeError(f"{model}: a forecast was fitted on or after its delivery day")
        if direct_pmf is not None:
            if not np.allclose(direct_grid, grid, atol=1e-7, rtol=0.0):
                raise RuntimeError(f"{model}: PMF grid does not match the common grid")
            component_pmf = normalize_pmf(direct_pmf)
        else:
            component_pmf = quantiles_to_discrete(
                quantiles, levels, p_zero, support=support, grid=grid
            ).pmf
        pmf[:, :, model_index, :] = component_pmf.astype(np.float32)
        cdf[:, :, model_index, :] = np.cumsum(component_pmf, axis=-1).astype(np.float32)

    split = config["split"]
    actual = curves.loc[days].to_numpy(dtype=float)
    burn_prices = curves.loc[split["burn_start"]:split["burn_end"]].to_numpy(dtype=float)
    regions = config["regions"]
    lower, upper = np.quantile(
        burn_prices, [float(regions["lower_quantile"]), float(regions["upper_quantile"])]
    )
    return PanelData(pmf, cdf, actual, grid, RegionSpec.from_thresholds(grid, float(lower), float(upper)))


def slice_panel(panel: PanelData, start: int, end: int) -> PanelData:
    return PanelData(
        panel.component_pmf[start:end],
        panel.component_cdf[start:end],
        panel.actual[start:end],
        panel.grid,
        panel.regions,
    )


def validation_objective(weights: Mapping[str, float], normalizers: Mapping[str, float]):
    """Weighted mean of validation scores, each divided by the Equal-pool score."""

    weights = {metric: float(value) for metric, value in weights.items() if float(value) > 0.0}
    total = float(sum(weights.values()))

    def evaluate(metrics: Mapping[str, float]) -> float:
        return float(
            sum(w * float(metrics[m]) / float(normalizers[m]) for m, w in weights.items()) / total
        )

    return evaluate


def refined_values(center: float, radius: float, step: float) -> list[float]:
    lower = max(-10.0, float(center) - float(radius))
    upper = min(10.0, float(center) + float(radius))
    return [float(value) for value in np.arange(lower, upper + 0.5 * step, step)]


def particle_specs(
    cfg: Mapping[str, Any],
    alpha1_values: Sequence[float],
    alpha2_values: Sequence[float],
) -> list[ParticleSpec]:
    return [
        ParticleSpec(
            particles=int(cfg["particles"]),
            state_noise=float(cfg["state_noise"]),
            coefficient_noise=float(cfg["coefficient_noise"]),
            resample_ess_fraction=float(cfg["resample_ess_fraction"]),
            alpha0=float(cfg["alpha0"]),
            alpha1=float(alpha1),
            alpha2=float(alpha2),
            random_seed=int(cfg["random_seed"]),
        )
        for alpha1 in alpha1_values
        for alpha2 in alpha2_values
    ]


def forecast_scores(
    method: str,
    days: pd.DatetimeIndex,
    panel: PanelData,
    result: CombinationResult,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Mean and daily CRPS, threshold-weighted CRPS and log score."""

    pmf = np.asarray(result.pmf, dtype=float)
    slot_scores = {
        "CRPS": discrete_crps(pmf, panel.actual, panel.grid),
        "twCRPS_L": discrete_lower_twcrps(pmf, panel.actual, panel.grid, panel.regions.lower_threshold),
        "twCRPS_U": discrete_upper_twcrps(pmf, panel.actual, panel.grid, panel.regions.upper_threshold),
        "Log_score": discrete_log_score(pmf, panel.actual, panel.grid),
    }
    daily = pd.DataFrame(
        {name: np.asarray(values, dtype=float).mean(axis=1) for name, values in slot_scores.items()}
    )
    daily.insert(0, "delivery_day", days.strftime("%Y-%m-%d"))
    daily.insert(0, "method", method)
    row = {"method": method, "n_days": int(panel.days)}
    row.update({name: float(np.mean(values)) for name, values in slot_scores.items()})
    return row, daily
