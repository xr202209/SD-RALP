"""Density combinations on a common price grid for the electricity application.

The caller supplies daily panels of component probability masses and CDFs.
Every dynamic method uses one weight vector per delivery day, shared by the 96
quarter-hour slots, and applies a day's realised scores only from the next day
on.  Probability masses can carry atoms at zero and at the price cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from distributions import (
    discrete_crps,
    discrete_log_score,
    discrete_lower_twcrps,
    discrete_upper_twcrps,
    quantiles_to_discrete,
)


EPS = 1e-12
CORE_METHODS = ("equal", "dynamic_global", "static_ralp", "sd_ralp")
# Ridge of the simplex-information score scaling.
SCORE_SCALING_RIDGE = 1e-6


def softmax(values: np.ndarray) -> np.ndarray:
    """Stable one-dimensional softmax."""

    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    exp_values = np.exp(shifted)
    return exp_values / max(float(exp_values.sum()), EPS)


def row_softmax(values: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax."""

    array = np.asarray(values, dtype=float)
    shifted = array - np.max(array, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), EPS)


@dataclass(frozen=True)
class RegionSpec:
    """Fixed lower/centre/upper partition on the common grid.

    ``lower_end_index`` and ``upper_end_index`` are inclusive.  Actual prices
    are classified with the supplied thresholds rather than rounded grid
    values, so thresholds may lie between grid points.
    """

    lower_threshold: float
    upper_threshold: float
    lower_end_index: int
    upper_end_index: int

    @classmethod
    def from_thresholds(
        cls,
        grid: np.ndarray,
        lower_threshold: float,
        upper_threshold: float,
    ) -> "RegionSpec":
        x = np.asarray(grid, dtype=float)
        if not float(lower_threshold) < float(upper_threshold):
            raise ValueError("lower_threshold must be smaller than upper_threshold")
        lower_end = int(np.searchsorted(x, float(lower_threshold), side="right") - 1)
        upper_end = int(np.searchsorted(x, float(upper_threshold), side="right") - 1)
        spec = cls(float(lower_threshold), float(upper_threshold), lower_end, upper_end)
        spec.validate(len(x))
        return spec

    @classmethod
    def from_indices(
        cls,
        grid: np.ndarray,
        lower_end_index: int,
        upper_end_index: int,
    ) -> "RegionSpec":
        x = np.asarray(grid, dtype=float)
        spec = cls(
            float(x[int(lower_end_index)]),
            float(x[int(upper_end_index)]),
            int(lower_end_index),
            int(upper_end_index),
        )
        spec.validate(len(x))
        return spec

    def validate(self, grid_size: int) -> None:
        if not (0 <= self.lower_end_index < self.upper_end_index < grid_size - 1):
            raise ValueError(
                "region indices must leave non-empty lower, centre, and upper grid regions"
            )
        if not self.lower_threshold < self.upper_threshold:
            raise ValueError("invalid region thresholds")

    def masks(self, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.validate(grid_size)
        index = np.arange(grid_size)
        return (
            index <= self.lower_end_index,
            (index > self.lower_end_index) & (index <= self.upper_end_index),
            index > self.upper_end_index,
        )

    def classify(self, actual: np.ndarray) -> np.ndarray:
        y = np.asarray(actual, dtype=float)
        return np.where(y <= self.lower_threshold, 0, np.where(y <= self.upper_threshold, 1, 2))


@dataclass(frozen=True)
class PanelData:
    """Daily component distributions on a common discrete price grid."""

    component_pmf: np.ndarray
    component_cdf: np.ndarray
    actual: np.ndarray
    grid: np.ndarray
    regions: RegionSpec

    def __post_init__(self) -> None:
        pmf = np.asarray(self.component_pmf)
        cdf = np.asarray(self.component_cdf)
        actual = np.asarray(self.actual)
        grid = np.asarray(self.grid)
        object.__setattr__(self, "component_pmf", pmf)
        object.__setattr__(self, "component_cdf", cdf)
        object.__setattr__(self, "actual", actual)
        object.__setattr__(self, "grid", grid)

        if pmf.ndim != 4:
            raise ValueError("component_pmf must have shape (days, 96, models, grid)")
        if cdf.shape != pmf.shape:
            raise ValueError("component_cdf must have the same shape as component_pmf")
        days, slots, models, grid_size = pmf.shape
        if slots != 96:
            raise ValueError(f"the Shanxi panel must contain 96 slots per day, got {slots}")
        if days < 1 or models < 1 or grid_size < 3:
            raise ValueError("panel dimensions must all be non-empty")
        if actual.shape != (days, slots):
            raise ValueError(f"actual must have shape {(days, slots)}, got {actual.shape}")
        if grid.shape != (grid_size,):
            raise ValueError(f"grid must have shape {(grid_size,)}, got {grid.shape}")
        if not np.all(np.diff(grid.astype(float)) > 0.0):
            raise ValueError("grid must be strictly increasing")
        if not np.isfinite(pmf).all() or not np.isfinite(cdf).all() or not np.isfinite(actual).all():
            raise ValueError("panel contains non-finite values")
        if float(np.min(pmf)) < -5e-7:
            raise ValueError("component PMFs contain negative mass")
        totals = pmf.sum(axis=-1)
        if not np.allclose(totals, 1.0, atol=5e-5, rtol=0.0):
            raise ValueError("each component PMF must sum to one")
        if float(np.min(np.diff(cdf, axis=-1))) < -5e-6:
            raise ValueError("component CDFs must be non-decreasing")
        if not np.allclose(cdf[..., -1], 1.0, atol=5e-5, rtol=0.0):
            raise ValueError("each component CDF must end at one")
        if not np.allclose(cdf, np.cumsum(np.maximum(pmf, 0.0), axis=-1), atol=8e-5, rtol=0.0):
            raise ValueError("component_cdf is inconsistent with component_pmf")
        self.regions.validate(grid_size)

    @property
    def days(self) -> int:
        return int(self.component_pmf.shape[0])

    @property
    def slots(self) -> int:
        return int(self.component_pmf.shape[1])

    @property
    def model_count(self) -> int:
        return int(self.component_pmf.shape[2])

    @property
    def grid_size(self) -> int:
        return int(self.component_pmf.shape[3])


def concatenate_panels(panels: Sequence[PanelData]) -> PanelData:
    """Concatenate chronological panels after checking their common design."""

    if not panels:
        raise ValueError("at least one panel is required")
    first = panels[0]
    for panel in panels[1:]:
        if panel.model_count != first.model_count or not np.array_equal(panel.grid, first.grid):
            raise ValueError("panels do not share the same models/grid")
        if panel.regions != first.regions:
            raise ValueError("panels do not share the same regional partition")
    return PanelData(
        component_pmf=np.concatenate([panel.component_pmf for panel in panels], axis=0),
        component_cdf=np.concatenate([panel.component_cdf for panel in panels], axis=0),
        actual=np.concatenate([panel.actual for panel in panels], axis=0),
        grid=first.grid,
        regions=first.regions,
    )


def actual_grid_indices(grid: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Map realised prices to their nearest grid cell using midpoint edges."""

    x = np.asarray(grid, dtype=float)
    y = np.asarray(actual, dtype=float)
    edges = 0.5 * (x[:-1] + x[1:])
    return np.searchsorted(edges, y, side="right").astype(np.int64)


def component_mass_at_actual(data: PanelData) -> np.ndarray:
    """Return component probability mass in each realised price cell."""

    indices = actual_grid_indices(data.grid, data.actual)
    gather = np.broadcast_to(
        indices[:, :, None, None],
        (data.days, data.slots, data.model_count, 1),
    )
    return np.take_along_axis(data.component_pmf, gather, axis=-1)[..., 0]


def daily_component_log_scores(data: PanelData) -> np.ndarray:
    """Daily mean log probability mass for each component (days, models)."""

    return np.mean(np.log(np.maximum(component_mass_at_actual(data), EPS)), axis=1)


@dataclass(frozen=True)
class RegionalStats:
    """Precomputed auxiliary CRPS terms used by static and GAS weights."""

    losses: np.ndarray
    disagreement: np.ndarray
    observed_region: np.ndarray

    @property
    def days(self) -> int:
        return int(self.losses.shape[0])

    @property
    def model_count(self) -> int:
        return int(self.losses.shape[-1])


def compute_regional_stats(data: PanelData) -> RegionalStats:
    """Regional losses ``b`` and pairwise CDF distances ``D`` of the components.

    The four scopes are ordered ``global, L, C, U``.  The integrals are exact for
    the piecewise-constant CDFs, so the thresholds need not lie on the grid.
    """

    cdf = np.asarray(data.component_cdf, dtype=float)
    cdf_left = cdf[..., :-1]
    interval_grid_left = data.grid[:-1]
    interval_grid_right = data.grid[1:]
    bounds = (
        (float(data.grid[0]), float(data.grid[-1])),
        (float(data.grid[0]), float(data.regions.lower_threshold)),
        (
            float(data.regions.lower_threshold),
            float(data.regions.upper_threshold),
        ),
        (float(data.regions.upper_threshold), float(data.grid[-1])),
    )
    losses = np.empty((data.days, data.slots, 4, data.model_count), dtype=float)
    disagreement = np.empty(
        (data.days, data.slots, 4, data.model_count, data.model_count),
        dtype=float,
    )
    actual = np.asarray(data.actual, dtype=float)
    for region_index, (region_lower, region_upper) in enumerate(bounds):
        interval_left = np.maximum(interval_grid_left, region_lower)
        interval_right = np.minimum(interval_grid_right, region_upper)
        interval_width = np.maximum(interval_right - interval_left, 0.0)

        # The discrete CDF is constant on each support interval.  Expanding
        # (F - 1{y <= z})^2 avoids a large model x interval indicator tensor.
        above_actual_width = np.maximum(
            interval_right[None, None, :]
            - np.maximum(
                interval_left[None, None, :],
                actual[..., None],
            ),
            0.0,
        )
        cdf_square_integral = np.einsum(
            "dskg,g->dsk",
            cdf_left * cdf_left,
            interval_width,
            optimize=True,
        )
        cdf_indicator_integral = np.einsum(
            "dskg,dsg->dsk",
            cdf_left,
            above_actual_width,
            optimize=True,
        )
        indicator_integral = above_actual_width.sum(axis=-1)
        losses[:, :, region_index, :] = np.maximum(
            cdf_square_integral
            - 2.0 * cdf_indicator_integral
            + indicator_integral[..., None],
            0.0,
        )
        cross = np.einsum(
            "dskg,g,dslg->dskl",
            cdf_left,
            interval_width,
            cdf_left,
            optimize=True,
        )
        diagonal = np.diagonal(cross, axis1=-2, axis2=-1)
        distance = diagonal[..., :, None] + diagonal[..., None, :] - 2.0 * cross
        disagreement[:, :, region_index, :, :] = np.maximum(distance, 0.0)
    if not np.allclose(
        losses[:, :, 0, :],
        losses[:, :, 1:, :].sum(axis=2),
        atol=1e-9,
        rtol=1e-9,
    ):
        raise RuntimeError("exact regional component losses do not sum to global CRPS")
    if not np.allclose(
        disagreement[:, :, 0, :, :],
        disagreement[:, :, 1:, :, :].sum(axis=2),
        atol=1e-9,
        rtol=1e-9,
    ):
        raise RuntimeError("exact regional disagreements do not sum to global distance")
    return RegionalStats(losses, disagreement, data.regions.classify(data.actual))


@dataclass(frozen=True)
class StaticWeights:
    global_weights: np.ndarray
    lower: np.ndarray
    center: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(value, dtype=float) for value in self.as_tuple()]
        size = arrays[0].shape
        if len(size) != 1 or any(value.shape != size for value in arrays):
            raise ValueError("all static weight vectors must have the same one-dimensional shape")
        for value in arrays:
            if float(np.min(value)) < -1e-10 or not np.isclose(float(value.sum()), 1.0, atol=1e-8):
                raise ValueError("static weights must lie on the simplex")
        object.__setattr__(self, "global_weights", arrays[0])
        object.__setattr__(self, "lower", arrays[1])
        object.__setattr__(self, "center", arrays[2])
        object.__setattr__(self, "upper", arrays[3])

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.global_weights, self.lower, self.center, self.upper

    def regional_matrix(self) -> np.ndarray:
        return np.vstack([self.lower, self.center, self.upper])

    @classmethod
    def equal(cls, model_count: int) -> "StaticWeights":
        weights = np.repeat(1.0 / int(model_count), int(model_count))
        return cls(weights.copy(), weights.copy(), weights.copy(), weights.copy())


def auxiliary_logit_score(
    losses: np.ndarray,
    disagreement: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Negative logit-gradient of the auxiliary regional CRPS loss."""

    b = np.asarray(losses, dtype=float)
    d = np.asarray(disagreement, dtype=float)
    w = np.asarray(weights, dtype=float)
    direct_gradient = b - np.einsum("...kl,l->...k", d, w, optimize=True)
    weighted_average = np.sum(direct_gradient * w, axis=-1, keepdims=True)
    return w * (weighted_average - direct_gradient)


def simplex_scaled_score(
    score: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Scale the score by the inverse square root of the simplex information."""

    values = np.asarray(score, dtype=float)
    w = np.asarray(weights, dtype=float)
    if values.shape != w.shape or values.ndim != 1:
        raise ValueError("score and weights must be matching one-dimensional vectors")
    if not np.isfinite(values).all() or not np.isfinite(w).all():
        raise ValueError("score scaling received non-finite values")
    if float(ridge) <= 0.0:
        raise ValueError("score-scaling ridge must be strictly positive")
    information = np.diag(w) - np.outer(w, w)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    inverse_sqrt = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, 0.0) + float(ridge))
    ) @ eigenvectors.T
    scaled = inverse_sqrt @ values
    # Numerical eigensolvers can introduce a tiny component in the unidentified
    # common-logit direction. Removing it leaves the softmax path unchanged.
    return scaled - float(np.mean(scaled))


def fit_static_weights(
    data: PanelData,
    eta: float,
    *,
    stats: RegionalStats | None = None,
) -> StaticWeights:
    """Static global and regional weights proportional to exp(-eta * mean loss)."""

    scoped = compute_regional_stats(data) if stats is None else stats
    if scoped.days != data.days or scoped.model_count != data.model_count:
        raise ValueError("regional statistics do not match the panel")
    fitted = []
    for stat_index in range(4):
        regional_losses = scoped.losses[:, :, stat_index, :].reshape(-1, data.model_count)
        mean_losses = np.mean(regional_losses, axis=0)
        fitted.append(softmax(-float(eta) * mean_losses))
    return StaticWeights(*fitted)


@dataclass
class CombinationResult:
    method: str
    pmf: np.ndarray
    cdf: np.ndarray
    weights_trace: np.ndarray
    normalizer: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self, data: PanelData) -> None:
        if self.pmf.shape != (data.days, data.slots, data.grid_size):
            raise ValueError("combined PMF has the wrong shape")
        if self.cdf.shape != self.pmf.shape:
            raise ValueError("combined CDF has the wrong shape")
        if self.weights_trace.shape != (data.days, 3, data.model_count):
            raise ValueError("weight trace has the wrong shape")
        if self.normalizer.shape != (data.days, data.slots):
            raise ValueError("normalizer has the wrong shape")
        if not np.isfinite(self.pmf).all() or float(np.min(self.pmf)) < -1e-9:
            raise ValueError("combined PMF is invalid")
        if not np.allclose(self.pmf.sum(axis=-1), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("combined PMF does not sum to one")
        if not np.allclose(self.cdf, np.cumsum(self.pmf, axis=-1), atol=1e-8, rtol=0.0):
            raise ValueError("combined CDF is inconsistent with PMF")
        if float(np.min(self.weights_trace)) < -1e-10:
            raise ValueError("negative combination weight")
        if not np.allclose(self.weights_trace.sum(axis=-1), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("weights do not sum to one")


def combine_regional_pmf(
    data: PanelData,
    weights_trace: np.ndarray,
    *,
    method: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> CombinationResult:
    """Normalised regional linear pool on the price grid."""

    weights = np.asarray(weights_trace, dtype=float)
    if weights.ndim == 2:
        if weights.shape != (3, data.model_count):
            raise ValueError("constant regional weights have the wrong shape")
        weights = np.broadcast_to(weights[None, :, :], (data.days, 3, data.model_count)).copy()
    if weights.shape != (data.days, 3, data.model_count):
        raise ValueError("weights_trace must have shape (days, 3, models)")
    if float(np.min(weights)) < -1e-10 or not np.allclose(
        weights.sum(axis=-1), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("regional weights must lie on the simplex")

    unnormalised = np.zeros((data.days, data.slots, data.grid_size), dtype=float)
    for region_index, mask in enumerate(data.regions.masks(data.grid_size)):
        unnormalised[:, :, mask] = np.einsum(
            "dk,dskm->dsm",
            weights[:, region_index, :],
            data.component_pmf[:, :, :, mask],
            optimize=True,
        )
    normalizer = unnormalised.sum(axis=-1)
    if np.any(normalizer <= EPS):
        raise RuntimeError("regional pool has zero total probability mass")
    pmf = unnormalised / normalizer[..., None]
    pmf = np.maximum(pmf, 0.0)
    pmf /= pmf.sum(axis=-1, keepdims=True)
    result = CombinationResult(
        method=method,
        pmf=pmf,
        cdf=np.cumsum(pmf, axis=-1),
        weights_trace=weights,
        normalizer=normalizer,
        diagnostics=dict(diagnostics or {}),
    )
    result.validate(data)
    return result


def run_constant_pool(data: PanelData, weights: np.ndarray, *, method: str) -> CombinationResult:
    """Run a constant global linear pool."""

    w = np.asarray(weights, dtype=float)
    if w.shape != (data.model_count,) or float(np.min(w)) < -1e-10:
        raise ValueError("invalid global weights")
    w = np.maximum(w, 0.0)
    w /= max(float(w.sum()), EPS)
    trace = np.broadcast_to(w[None, None, :], (data.days, 3, data.model_count)).copy()
    return combine_regional_pmf(data, trace, method=method)


def _mean_daily_score(
    stats: RegionalStats,
    day: int,
    stat_index: int,
    weights: np.ndarray,
) -> np.ndarray:
    raw_score = np.mean(
        auxiliary_logit_score(
            stats.losses[day, :, stat_index, :],
            stats.disagreement[day, :, stat_index, :, :],
            weights,
        ),
        axis=0,
    )
    return simplex_scaled_score(raw_score, weights, ridge=SCORE_SCALING_RIDGE)


def _check_warm_panel(score_data: PanelData, warm_data: PanelData | None) -> None:
    if warm_data is None:
        return
    if warm_data.model_count != score_data.model_count or not np.array_equal(
        warm_data.grid, score_data.grid
    ):
        raise ValueError("warm and score panels must share models and grid")
    if warm_data.regions != score_data.regions:
        raise ValueError("warm and score panels must share regions")


def run_core_method(
    score_data: PanelData,
    method: str,
    static_weights: StaticWeights,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    rho: float = 0.0,
    warm_data: PanelData | None = None,
    score_stats: RegionalStats | None = None,
    warm_stats: RegionalStats | None = None,
) -> CombinationResult:
    """Run one core method with one-day-delayed daily updating.

    Weight trace entry ``d`` is the weight used to forecast day ``d``.  The
    realised scores from day ``d`` are applied only after that entry has been
    recorded, and can therefore first affect trace entry ``d + 1``.
    """

    if method not in CORE_METHODS:
        raise ValueError(f"unknown core method {method!r}; expected one of {CORE_METHODS}")
    _check_warm_panel(score_data, warm_data)
    k = score_data.model_count
    if static_weights.global_weights.shape != (k,):
        raise ValueError("static weights do not match the score panel")

    if method == "equal":
        return run_constant_pool(score_data, np.repeat(1.0 / k, k), method="Equal pool")
    if method == "static_ralp":
        return combine_regional_pmf(
            score_data,
            static_weights.regional_matrix(),
            method="Static RALP",
        )

    scoped_score = compute_regional_stats(score_data) if score_stats is None else score_stats
    scoped_warm = None
    if warm_data is not None:
        scoped_warm = compute_regional_stats(warm_data) if warm_stats is None else warm_stats

    alpha = float(alpha)
    beta = float(beta)
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")

    if method == "dynamic_global":
        theta = np.log(static_weights.global_weights + EPS)
        omega = (1.0 - beta) * theta.copy()

        def process_global(stats: RegionalStats, collect: bool) -> list[np.ndarray]:
            nonlocal theta
            trace: list[np.ndarray] = []
            for day in range(stats.days):
                weights = softmax(theta)
                if collect:
                    trace.append(np.vstack([weights, weights, weights]))
                score = _mean_daily_score(stats, day, 0, weights)
                theta = np.clip(omega + beta * theta + alpha * score, -7.0, 7.0)
            return trace

        if scoped_warm is not None:
            process_global(scoped_warm, collect=False)
        weight_trace = np.asarray(process_global(scoped_score, collect=True))
        return combine_regional_pmf(
            score_data,
            weight_trace,
            method="Global GAS",
            diagnostics={
                "alpha": alpha,
                "beta": beta,
                "one_day_delayed": True,
            },
        )

    anchor_c = np.log(static_weights.center + EPS)
    anchor_l = np.log((1.0 - rho) * static_weights.lower + rho * static_weights.center + EPS)
    anchor_u = np.log((1.0 - rho) * static_weights.upper + rho * static_weights.center + EPS)
    theta_c = anchor_c.copy()
    theta_l = anchor_l.copy()
    theta_u = anchor_u.copy()

    def process_sd(stats: RegionalStats, collect: bool) -> list[np.ndarray]:
        nonlocal theta_l, theta_c, theta_u
        trace: list[np.ndarray] = []
        for day in range(stats.days):
            w_l, w_c, w_u = softmax(theta_l), softmax(theta_c), softmax(theta_u)
            if collect:
                trace.append(np.vstack([w_l, w_c, w_u]))

            center_score = _mean_daily_score(stats, day, 2, w_c)
            lower_score = _mean_daily_score(stats, day, 1, w_l)
            upper_score = _mean_daily_score(stats, day, 3, w_u)
            theta_c = np.clip(
                anchor_c + beta * (theta_c - anchor_c) + alpha * center_score,
                -7.0,
                7.0,
            )
            theta_l = np.clip(
                anchor_l
                + (1.0 - rho) * beta * (theta_l - anchor_l)
                + (1.0 - rho) * alpha * lower_score
                + rho * (theta_c - anchor_c),
                -7.0,
                7.0,
            )
            theta_u = np.clip(
                anchor_u
                + (1.0 - rho) * beta * (theta_u - anchor_u)
                + (1.0 - rho) * alpha * upper_score
                + rho * (theta_c - anchor_c),
                -7.0,
                7.0,
            )
        return trace

    if scoped_warm is not None:
        process_sd(scoped_warm, collect=False)
    weight_trace = np.asarray(process_sd(scoped_score, collect=True))
    return combine_regional_pmf(
        score_data,
        weight_trace,
        method="SD-RALP",
        diagnostics={
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "one_day_delayed": True,
        },
    )


def score_combination(data: PanelData, result: CombinationResult) -> dict[str, float]:
    """Mean scores of a combined forecast, used by the validation criterion."""

    result.validate(data)
    crps = np.asarray(discrete_crps(result.pmf, data.actual, data.grid), dtype=float)
    lower = np.asarray(
        discrete_lower_twcrps(
            result.pmf,
            data.actual,
            data.grid,
            data.regions.lower_threshold,
        ),
        dtype=float,
    )
    upper = np.asarray(
        discrete_upper_twcrps(
            result.pmf,
            data.actual,
            data.grid,
            data.regions.upper_threshold,
        ),
        dtype=float,
    )
    center = crps - lower - upper
    lower_mask, _, upper_mask = data.regions.masks(data.grid_size)
    p_lower = np.asarray(result.pmf[..., lower_mask].sum(axis=-1), dtype=float)
    p_upper = np.asarray(result.pmf[..., upper_mask].sum(axis=-1), dtype=float)
    event_lower = (data.actual <= data.regions.lower_threshold).astype(float)
    event_upper = (data.actual > data.regions.upper_threshold).astype(float)
    metrics = {
        "CRPS": float(np.mean(crps)),
        "twCRPS_L": float(np.mean(lower)),
        "twCRPS_C": float(np.mean(center)),
        "twCRPS_U": float(np.mean(upper)),
        "Brier_L": float(np.mean(np.square(p_lower - event_lower))),
        "Brier_U": float(np.mean(np.square(p_upper - event_upper))),
        "ECAL_L": float(abs(np.mean(p_lower) - np.mean(event_lower))),
        "ECAL_U": float(abs(np.mean(p_upper) - np.mean(event_upper))),
        "forecast_rate_L": float(np.mean(p_lower)),
        "forecast_rate_U": float(np.mean(p_upper)),
        "observed_rate_L": float(np.mean(event_lower)),
        "observed_rate_U": float(np.mean(event_upper)),
        "Log_score": float(
            discrete_log_score(
                result.pmf,
                data.actual,
                data.grid,
                min_probability=EPS,
                reduction="mean",
            )
        ),
    }
    return metrics


@dataclass(frozen=True)
class CoreTuningResult:
    method: str
    parameters: dict[str, float]
    static_weights: StaticWeights
    rows: tuple[dict[str, float | bool | str], ...]


def tune_core_method(
    train: PanelData,
    validation: PanelData,
    method: str,
    *,
    eta_grid: Sequence[float],
    alpha_grid: Sequence[float] = (0.0,),
    beta_grid: Sequence[float] = (0.0,),
    rho_grid: Sequence[float] = (0.0,),
    objective: Callable[[Mapping[str, float]], float] | None = None,
    train_stats: RegionalStats | None = None,
    validation_stats: RegionalStats | None = None,
) -> CoreTuningResult:
    """Grid search of a core method on the validation panel."""

    if method not in {"dynamic_global", "static_ralp", "sd_ralp"}:
        raise ValueError("only fitted core methods can be tuned")
    _check_warm_panel(validation, train)
    train_scoped = compute_regional_stats(train) if train_stats is None else train_stats
    val_scoped = compute_regional_stats(validation) if validation_stats is None else validation_stats
    objective_fn = objective or (lambda metrics: float(metrics["CRPS"]))
    rows: list[dict[str, float | bool | str]] = []
    best: tuple[float, dict[str, float], StaticWeights] | None = None

    for eta in [float(value) for value in eta_grid]:
        static = fit_static_weights(
            train,
            eta,
            stats=train_scoped,
        )
        if method == "static_ralp":
            candidates = [(0.0, 0.0, 0.0)]
        elif method == "dynamic_global":
            candidates = [(float(a), float(b), 0.0) for a, b in product(alpha_grid, beta_grid)]
        else:
            candidates = [
                (float(a), float(b), float(r))
                for a, b, r in product(alpha_grid, beta_grid, rho_grid)
            ]
        for alpha, beta, rho in candidates:
            result = run_core_method(
                validation,
                method,
                static,
                alpha=alpha,
                beta=beta,
                rho=rho,
                warm_data=train if method in {"dynamic_global", "sd_ralp"} else None,
                score_stats=val_scoped,
                warm_stats=train_scoped,
            )
            metrics = score_combination(validation, result)
            value = float(objective_fn(metrics))
            row: dict[str, float | bool | str] = {
                "method": method,
                "eta": eta,
                "alpha": alpha,
                "beta": beta,
                "rho": rho,
                "objective": value,
                **metrics,
                "selected": False,
            }
            rows.append(row)
            params = {"eta": eta, "alpha": alpha, "beta": beta, "rho": rho}
            if best is None or value < best[0]:
                best = (value, params, static)
    if best is None:
        raise RuntimeError("empty tuning grid")
    for row in rows:
        row["selected"] = all(float(row[key]) == float(best[1][key]) for key in best[1])
    return CoreTuningResult(method, best[1], best[2], tuple(rows))


def fit_log_score_pool(train: PanelData) -> tuple[np.ndarray, str]:
    """Static log-score-optimal linear pool fitted on training outcomes."""

    mass = component_mass_at_actual(train).reshape(-1, train.model_count)
    k = train.model_count

    def objective(weights: np.ndarray) -> float:
        return float(-np.mean(np.log(np.maximum(mass @ weights, EPS))))

    result = minimize(
        objective,
        np.repeat(1.0 / k, k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"log-score pool optimisation failed: {result.message}")
    weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    weights /= max(float(weights.sum()), EPS)
    return weights, str(result.message)


def fit_bma_weights(train: PanelData) -> np.ndarray:
    """Training-posterior BMA weights from summed daily log scores."""

    return softmax(np.sum(daily_component_log_scores(train), axis=0))


def _discrete_quantile(cdf: np.ndarray, grid: np.ndarray, probability: float) -> np.ndarray:
    indices = np.argmax(cdf >= float(probability), axis=-1)
    return np.asarray(grid)[indices]


def run_quantile_average_pool(
    score_data: PanelData,
    *,
    quantile_points: int = 200,
    probability_range: tuple[float, float] | None = None,
) -> CombinationResult:
    """Equal-weight average on a fixed inverse-CDF probability grid."""

    if int(quantile_points) != quantile_points or quantile_points < 2:
        raise ValueError("quantile_points must be an integer of at least two")
    quantile_points = int(quantile_points)
    if probability_range is None:
        probabilities = (np.arange(quantile_points, dtype=float) + 0.5) / quantile_points
        probability_grid = "midpoint"
    else:
        probability_min, probability_max = map(float, probability_range)
        if not 0.0 < probability_min < probability_max < 1.0:
            raise ValueError("probability_range must be strictly inside (0, 1)")
        probabilities = np.linspace(probability_min, probability_max, quantile_points)
        probability_grid = "fixed_range"
    component_quantiles = np.stack(
        [
            _discrete_quantile(score_data.component_cdf, score_data.grid, probability)
            for probability in probabilities
        ],
        axis=-1,
    )
    averaged_quantiles = np.mean(component_quantiles, axis=2)
    # For non-negative prices, the quantile average is zero only while every
    # component quantile is zero; hence its zero atom is the smallest component
    # zero atom. The midpoint cells each carry probability 1/quantile_points.
    p_zero = np.min(score_data.component_pmf[..., 0], axis=2)
    distribution = quantiles_to_discrete(
        averaged_quantiles,
        probabilities,
        p_zero,
        support=(float(score_data.grid[0]), float(score_data.grid[-1])),
        grid=score_data.grid,
    )
    pmf = distribution.pmf
    cdf = distribution.cdf
    equal = np.repeat(1.0 / score_data.model_count, score_data.model_count)
    trace = np.broadcast_to(
        equal[None, None, :],
        (score_data.days, 3, score_data.model_count),
    ).copy()
    result = CombinationResult(
        method="Quantile-average pool",
        pmf=pmf,
        cdf=cdf,
        weights_trace=trace,
        normalizer=np.ones((score_data.days, score_data.slots)),
        diagnostics={
            "quantile_average": f"equal-weight average on {quantile_points} fixed probability points",
            "quantile_points": quantile_points,
            "probability_levels": probabilities.tolist(),
            "probability_grid": probability_grid,
            "reconstruction": f"{quantile_points} probability cells on common price grid",
        },
    )
    result.validate(score_data)
    return result


@dataclass(frozen=True)
class ParticleSpec:
    particles: int = 1000
    state_noise: float = 0.12
    coefficient_noise: float = 0.06
    resample_ess_fraction: float = 0.50
    alpha0: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0
    random_seed: int = 20260717

    def __post_init__(self) -> None:
        if int(self.particles) < 2:
            raise ValueError("particles must be at least two")
        if self.state_noise < 0.0 or self.coefficient_noise < 0.0:
            raise ValueError("particle noise scales must be non-negative")
        if not 0.0 < self.resample_ess_fraction <= 1.0:
            raise ValueError("resample_ess_fraction must lie in (0,1]")


@dataclass
class _ParticleState:
    x: np.ndarray
    alpha: np.ndarray
    log_weights: np.ndarray
    rng: np.random.Generator


def _initial_particle_state(spec: ParticleSpec, model_count: int) -> _ParticleState:
    return _ParticleState(
        x=np.zeros((int(spec.particles), int(model_count)), dtype=float),
        alpha=np.tile(
            np.asarray([spec.alpha0, spec.alpha1, spec.alpha2], dtype=float),
            (int(spec.particles), 1),
        ),
        log_weights=np.repeat(-np.log(float(spec.particles)), int(spec.particles)),
        rng=np.random.default_rng(int(spec.random_seed)),
    )


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _panel_diversity(data: PanelData) -> np.ndarray:
    """Daily DTVW diversity vector based on component distribution means."""

    means = np.einsum("dskg,g->dsk", data.component_pmf, data.grid, optimize=True)
    output = np.empty((data.days, data.model_count), dtype=float)
    for day in range(data.days):
        pairwise = means[day, :, :, None] - means[day, :, None, :]
        squared = pairwise * pairwise
        denominator = np.sum(squared, axis=(1, 2))
        vectors = np.zeros((data.slots, data.model_count), dtype=float)
        valid = denominator > EPS
        if np.any(valid):
            vectors[valid] = np.sum(squared[valid], axis=2) / denominator[valid, None]
        output[day] = np.mean(vectors, axis=0)
    return output


def run_particle_pool(
    train: PanelData,
    score_data: PanelData,
    spec: ParticleSpec,
    *,
    use_diversity: bool,
    warm_data: PanelData | None = None,
) -> CombinationResult:
    """TVW or DTVW particle-filter combination.

    The particle likelihood is the mean log pooled probability mass across the
    96 slots.  It is evaluated only after the day's effective weights have been
    used for prediction, so its first possible effect is the next day.
    """

    _check_warm_panel(score_data, train)
    _check_warm_panel(score_data, warm_data)
    state = _initial_particle_state(spec, score_data.model_count)
    traces: list[np.ndarray] = []
    ess_trace: list[float] = []

    def process(panel: PanelData, collect: bool) -> None:
        diversity = _panel_diversity(panel) if use_diversity else np.zeros(
            (panel.days, panel.model_count), dtype=float
        )
        realised_mass = component_mass_at_actual(panel)
        for day in range(panel.days):
            state.alpha += state.rng.normal(
                0.0,
                float(spec.coefficient_noise),
                size=state.alpha.shape,
            )
            coefficients = 2.0 * (
                1.0 / (1.0 + np.exp(-np.clip(state.alpha, -30.0, 30.0))) - 0.5
            )
            state.x = (
                coefficients[:, 0, None]
                + coefficients[:, 1, None] * state.x
                + coefficients[:, 2, None] * diversity[day][None, :]
                + state.rng.normal(0.0, float(spec.state_noise), size=state.x.shape)
            )
            state.x = np.clip(state.x, -12.0, 12.0)
            component_weights = row_softmax(state.x)
            particle_weights = softmax(state.log_weights)
            effective_weights = particle_weights @ component_weights
            if collect:
                traces.append(np.vstack([effective_weights, effective_weights, effective_weights]))

            pooled_mass = realised_mass[day] @ component_weights.T
            state.log_weights += np.mean(np.log(np.maximum(pooled_mass, EPS)), axis=0)
            state.log_weights -= float(np.max(state.log_weights))
            particle_weights = softmax(state.log_weights)
            ess_fraction = 1.0 / (
                float(spec.particles) * float(np.sum(particle_weights * particle_weights))
            )
            if collect:
                ess_trace.append(float(ess_fraction))
            if ess_fraction < float(spec.resample_ess_fraction):
                indices = _systematic_resample(particle_weights, state.rng)
                state.x = state.x[indices]
                state.alpha = state.alpha[indices]
                state.log_weights = np.repeat(-np.log(float(spec.particles)), int(spec.particles))

    process(train, collect=False)
    if warm_data is not None:
        process(warm_data, collect=False)
    process(score_data, collect=True)
    trace = np.asarray(traces)
    label = "DTVW" if use_diversity else "TVW"
    return combine_regional_pmf(
        score_data,
        trace,
        method=label,
        diagnostics={
            "particle_spec": spec,
            "use_diversity": bool(use_diversity),
            "one_day_delayed": True,
            "ess_fraction": np.asarray(ess_trace),
        },
    )


@dataclass(frozen=True)
class ParticleTuningResult:
    spec: ParticleSpec
    rows: tuple[dict[str, float | bool], ...]


def tune_particle_pool(
    train: PanelData,
    validation: PanelData,
    specs: Iterable[ParticleSpec],
    *,
    use_diversity: bool,
    objective: Callable[[Mapping[str, float]], float] | None = None,
) -> ParticleTuningResult:
    """Select the particle-filter specification by the validation criterion."""

    objective_fn = objective or (lambda metrics: float(metrics["CRPS"]))
    rows: list[dict[str, float | bool]] = []
    best: tuple[float, ParticleSpec] | None = None
    for spec in specs:
        result = run_particle_pool(train, validation, spec, use_diversity=use_diversity)
        metrics = score_combination(validation, result)
        value = float(objective_fn(metrics))
        rows.append(
            {
                "alpha0": float(spec.alpha0),
                "alpha1": float(spec.alpha1),
                "alpha2": float(spec.alpha2),
                "objective": value,
                **metrics,
                "selected": False,
            }
        )
        if best is None or value < best[0]:
            best = (value, spec)
    if best is None:
        raise RuntimeError("empty particle tuning grid")
    for row in rows:
        row["selected"] = (
            float(row["alpha0"]) == float(best[1].alpha0)
            and float(row["alpha1"]) == float(best[1].alpha1)
            and float(row["alpha2"]) == float(best[1].alpha2)
        )
    return ParticleTuningResult(best[1], tuple(rows))
