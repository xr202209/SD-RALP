"""Discrete predictive distributions on a common price grid.

Component forecasts given as 99 quantiles and a zero-price probability are
mapped to probability masses on a finite grid that keeps the zero price (and
the price cap) as exact endpoint atoms.

The conversion treats each supplied quantile as representing its probability
cell (bounded by midpoints of adjacent quantile levels).  The part of every
cell below ``p_zero`` is replaced by the exact zero atom; the remaining mass is
placed on the common grid, with linear splitting between neighbouring grid
points to reduce rounding bias.  Quantiles at the market cap are therefore
retained at the cap, including the unobserved probability cell above the last
reported quantile level.

All public scoring functions accept arbitrary leading dimensions.  The final
axis of a PMF/CDF is always the grid axis, and the final axis of a quantile
array is always the quantile-level axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]
Reduction = Literal["none", "mean", "sum"]

DEFAULT_SUPPORT = (0.0, 1500.0)
DEFAULT_GRID_STEP = 5.0
_ATOL = 1e-10


@dataclass(frozen=True)
class DiscreteDistribution:
    """A batch of distributions on one shared grid.

    Attributes
    ----------
    grid
        One-dimensional, strictly increasing support grid.
    pmf, cdf
        Arrays with shape ``quantiles.shape[:-1] + (len(grid),)``.
    p_zero
        Exact mass stored at ``grid[0]``.  This equals the supplied explicit
        zero probability (up to floating-point normalization tolerance).
    p_cap
        Probability in the final grid bin, including any explicit mass implied
        by quantiles equal to the market cap.
    repaired_quantiles
        Finite, support-clipped, monotonically nondecreasing source quantiles
        used during conversion.
    quantile_levels
        The strictly increasing probability levels of the source quantiles.
    """

    grid: np.ndarray
    pmf: np.ndarray
    cdf: np.ndarray
    p_zero: np.ndarray
    p_cap: np.ndarray
    repaired_quantiles: np.ndarray
    quantile_levels: np.ndarray


def make_grid(
    support: tuple[float, float] = DEFAULT_SUPPORT,
    step: float = DEFAULT_GRID_STEP,
) -> np.ndarray:
    """Return a common grid that contains both support endpoints exactly."""

    lower, upper = (float(support[0]), float(support[1]))
    step = float(step)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("support must contain two finite, increasing endpoints")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("grid step must be finite and positive")

    count = int(np.floor((upper - lower) / step))
    grid = lower + step * np.arange(count + 1, dtype=float)
    if grid[-1] < upper - _ATOL:
        grid = np.append(grid, upper)
    else:
        grid[-1] = upper
    return grid


def _validate_grid(grid: ArrayLike) -> np.ndarray:
    out = np.asarray(grid, dtype=float)
    if out.ndim != 1 or out.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least two points")
    if not np.isfinite(out).all() or not np.all(np.diff(out) > 0.0):
        raise ValueError("grid must be finite and strictly increasing")
    return out


def _validate_levels(levels: ArrayLike, expected_size: int | None = None) -> np.ndarray:
    out = np.asarray(levels, dtype=float)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("quantile levels must be a non-empty one-dimensional array")
    if expected_size is not None and out.size != expected_size:
        raise ValueError(
            f"quantile axis has size {expected_size}, but {out.size} levels were supplied"
        )
    if not np.isfinite(out).all() or np.any(out <= 0.0) or np.any(out >= 1.0):
        raise ValueError("quantile levels must be finite and strictly between zero and one")
    if not np.all(np.diff(out) > 0.0):
        raise ValueError("quantile levels must be strictly increasing")
    return out


def _broadcast_observations(observations: ArrayLike | float, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(observations, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite")
    try:
        return np.broadcast_to(values, shape)
    except ValueError as exc:
        raise ValueError(f"observations with shape {values.shape} cannot broadcast to {shape}") from exc


def _reduce(values: np.ndarray, reduction: Reduction) -> np.ndarray | float:
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "sum":
        return float(np.sum(values))
    raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")


def normalize_pmf(pmf: np.ndarray, *, atol: float = _ATOL) -> np.ndarray:
    """Validate and normalize PMFs along their final axis."""

    values = np.asarray(pmf, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("pmf must have a final grid axis with at least two points")
    if not np.isfinite(values).all():
        raise ValueError("pmf contains NaN or infinite values")
    if np.any(values < -atol):
        raise ValueError("pmf contains negative probabilities")
    values = np.maximum(values, 0.0)
    total = values.sum(axis=-1, keepdims=True)
    if np.any(total <= atol):
        raise ValueError("each pmf row must have positive total mass")
    return values / total


def _quantile_cell_bounds(levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probability cells represented by each reported quantile.

    For levels .01, ..., .99 the endpoint cells have width .015 and all
    interior cells have width .01.  Consequently, a reported q99 at the price
    cap retains the probability interval from .985 through 1.0 at that cap.
    """

    midpoints = 0.5 * (levels[:-1] + levels[1:])
    lower = np.concatenate(([0.0], midpoints))
    upper = np.concatenate((midpoints, [1.0]))
    return lower, upper


def quantiles_to_discrete(
    quantiles: np.ndarray,
    quantile_levels: ArrayLike,
    p_zero: np.ndarray | float,
    *,
    support: tuple[float, float] = DEFAULT_SUPPORT,
    grid: ArrayLike | None = None,
    grid_step: float = DEFAULT_GRID_STEP,
    repair_monotonic: bool = True,
) -> DiscreteDistribution:
    """Convert quantiles and an explicit zero atom to a common-grid PMF/CDF.

    Parameters
    ----------
    quantiles
        Array with quantile levels on its final axis.
    quantile_levels
        Strictly increasing levels in ``(0, 1)``.
    p_zero
        Explicit zero-price probability, scalar or broadcastable to the
        leading shape of ``quantiles``.
    support, grid, grid_step
        The grid defaults to 0, 5, ..., 1500.  A custom grid must contain the
        specified support endpoints; the explicit atom is stored at its first
        point and the cap mass at its last point.
    repair_monotonic
        If true, crossing input quantiles are repaired by cumulative maximum.
        If false, a crossing raises ``ValueError``.

    Notes
    -----
    The returned first-bin mass is exactly ``p_zero``.  Any positive
    probability cell whose reported quantile is still at zero because of level
    discretization is placed at the first *positive* grid point, never added to
    the zero atom.
    """

    raw = np.asarray(quantiles, dtype=float)
    if raw.ndim < 1 or raw.shape[-1] == 0:
        raise ValueError("quantiles must have a non-empty final quantile axis")
    if not np.isfinite(raw).all():
        raise ValueError("quantiles contain NaN or infinite values")
    levels = _validate_levels(quantile_levels, raw.shape[-1])

    lower_support, upper_support = float(support[0]), float(support[1])
    if not np.isfinite(lower_support) or not np.isfinite(upper_support) or upper_support <= lower_support:
        raise ValueError("support must contain two finite, increasing endpoints")
    common_grid = make_grid(support, grid_step) if grid is None else _validate_grid(grid)
    if not np.isclose(common_grid[0], lower_support, atol=_ATOL, rtol=0.0):
        raise ValueError("the first grid point must equal the lower support endpoint")
    if not np.isclose(common_grid[-1], upper_support, atol=_ATOL, rtol=0.0):
        raise ValueError("the final grid point must equal the upper support endpoint")

    clipped = np.clip(raw, lower_support, upper_support)
    if not repair_monotonic and np.any(np.diff(clipped, axis=-1) < -_ATOL):
        raise ValueError("input quantiles cross; enable repair_monotonic to repair them")
    repaired = np.maximum.accumulate(clipped, axis=-1)

    leading_shape = repaired.shape[:-1]
    zero = np.asarray(p_zero, dtype=float)
    if not np.isfinite(zero).all() or np.any(zero < 0.0) or np.any(zero > 1.0):
        raise ValueError("p_zero must be finite and lie in [0, 1]")
    try:
        zero = np.broadcast_to(zero, leading_shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(f"p_zero with shape {zero.shape} cannot broadcast to {leading_shape}") from exc

    cell_lower, cell_upper = _quantile_cell_bounds(levels)
    # Intersection of each quantile cell with (p_zero, 1].  These weights sum
    # to 1-p_zero for every distribution, including non-grid-aligned p_zero.
    positive_weights = np.maximum(
        0.0,
        cell_upper - np.maximum(cell_lower, zero[..., None]),
    )

    rows = int(np.prod(leading_shape, dtype=int)) if leading_shape else 1
    nq = repaired.shape[-1]
    repaired_flat = repaired.reshape(rows, nq)
    weight_flat = positive_weights.reshape(rows, nq)
    zero_flat = zero.reshape(rows)
    pmf_flat = np.zeros((rows, common_grid.size), dtype=float)
    pmf_flat[:, 0] = zero_flat

    # Positive mass cannot be assigned to the explicit zero atom.  Exact cap
    # quantiles remain exact and are assigned entirely to the final grid point.
    mapped = repaired_flat.copy()
    mapped = np.where(weight_flat > 0.0, np.maximum(mapped, common_grid[1]), mapped)
    mapped = np.clip(mapped, common_grid[1], common_grid[-1])

    upper_index = np.searchsorted(common_grid, mapped, side="right")
    upper_index = np.clip(upper_index, 1, common_grid.size - 1)
    lower_index = upper_index - 1
    at_cap = mapped >= common_grid[-1] - _ATOL
    lower_index = np.where(at_cap, common_grid.size - 1, lower_index)
    upper_index = np.where(at_cap, common_grid.size - 1, upper_index)

    lower_value = common_grid[lower_index]
    upper_value = common_grid[upper_index]
    denominator = upper_value - lower_value
    upper_fraction = np.divide(
        mapped - lower_value,
        denominator,
        out=np.zeros_like(mapped),
        where=denominator > 0.0,
    )
    upper_fraction = np.clip(upper_fraction, 0.0, 1.0)

    row_index = np.repeat(np.arange(rows), nq)
    np.add.at(
        pmf_flat,
        (row_index, lower_index.ravel()),
        (weight_flat * (1.0 - upper_fraction)).ravel(),
    )
    np.add.at(
        pmf_flat,
        (row_index, upper_index.ravel()),
        (weight_flat * upper_fraction).ravel(),
    )

    # Roundoff correction is assigned to the first positive point, not zero,
    # so pmf[..., 0] continues to represent exactly the supplied atom.
    residual = 1.0 - pmf_flat.sum(axis=1)
    pmf_flat[:, 1] += residual
    if np.any(pmf_flat < -1e-8):
        raise RuntimeError("quantile-to-PMF conversion produced negative mass")
    pmf_flat = np.maximum(pmf_flat, 0.0)
    # Renormalize positive mass separately to preserve p_zero exactly.
    positive_total = pmf_flat[:, 1:].sum(axis=1)
    target_positive = 1.0 - zero_flat
    active = target_positive > _ATOL
    pmf_flat[active, 1:] *= (
        target_positive[active] / np.maximum(positive_total[active], _ATOL)
    )[:, None]
    pmf_flat[~active, 1:] = 0.0
    pmf_flat[:, 0] = zero_flat

    pmf = pmf_flat.reshape(leading_shape + (common_grid.size,))
    cdf = np.cumsum(pmf, axis=-1)
    cdf = np.maximum.accumulate(cdf, axis=-1)
    cdf[..., -1] = 1.0
    return DiscreteDistribution(
        grid=common_grid,
        pmf=pmf,
        cdf=cdf,
        p_zero=pmf[..., 0].copy(),
        p_cap=pmf[..., -1].copy(),
        repaired_quantiles=repaired,
        quantile_levels=levels,
    )


def pmf_to_quantiles(
    pmf: np.ndarray,
    quantile_levels: ArrayLike,
    grid: ArrayLike,
) -> np.ndarray:
    """Return generalized-inverse quantiles from common-grid PMFs."""

    values = normalize_pmf(pmf)
    common_grid = _validate_grid(grid)
    if values.shape[-1] != common_grid.size:
        raise ValueError("pmf grid axis and grid length do not match")
    levels = np.asarray(quantile_levels, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("quantile levels must be a non-empty one-dimensional array")
    if not np.isfinite(levels).all() or np.any(levels < 0.0) or np.any(levels > 1.0):
        raise ValueError("requested quantile levels must lie in [0, 1]")
    if not np.all(np.diff(levels) >= 0.0):
        raise ValueError("requested quantile levels must be nondecreasing")

    cdf = np.cumsum(values, axis=-1)
    cdf[..., -1] = 1.0
    flat = cdf.reshape(-1, common_grid.size)
    result = np.empty((flat.shape[0], levels.size), dtype=float)
    # np.searchsorted has no axis argument, so loop over the forecast rows.
    for row_index, row in enumerate(flat):
        indices = np.searchsorted(row, levels, side="left")
        indices = np.clip(indices, 0, common_grid.size - 1)
        result[row_index] = common_grid[indices]
    return result.reshape(values.shape[:-1] + (levels.size,))


def _integrated_cdf_error(
    pmf: np.ndarray,
    observations: ArrayLike | float,
    grid: ArrayLike,
    *,
    region_lower: float | None,
    region_upper: float | None,
) -> np.ndarray:
    """Integrate squared CDF error over a bounded threshold region."""

    values = normalize_pmf(pmf)
    common_grid = _validate_grid(grid)
    if values.shape[-1] != common_grid.size:
        raise ValueError("pmf grid axis and grid length do not match")
    obs = _broadcast_observations(observations, values.shape[:-1])

    lower = float(common_grid[0] if region_lower is None else region_lower)
    upper = float(common_grid[-1] if region_upper is None else region_upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError("twCRPS region bounds must be finite and nondecreasing")

    interval_left = np.maximum(common_grid[:-1], lower)
    interval_right = np.minimum(common_grid[1:], upper)
    width = np.maximum(interval_right - interval_left, 0.0)
    cdf_left = np.cumsum(values, axis=-1)[..., :-1]

    y = obs[..., None]
    left = interval_left
    right = interval_right
    below_observation = y >= right
    above_observation = y <= left
    split_left = np.clip(y - left, 0.0, width)
    split_right = width - split_left
    contribution = np.where(
        below_observation,
        np.square(cdf_left) * width,
        np.where(
            above_observation,
            np.square(cdf_left - 1.0) * width,
            np.square(cdf_left) * split_left
            + np.square(cdf_left - 1.0) * split_right,
        ),
    )
    score = contribution.sum(axis=-1)

    # Complete the integral if the requested region or observation lies beyond
    # the finite market support.  Below support F=0; above support F=1.
    below_right = min(upper, common_grid[0])
    if below_right > lower:
        score += np.maximum(0.0, below_right - np.maximum(lower, obs))
    above_left = max(lower, common_grid[-1])
    if upper > above_left:
        score += np.maximum(0.0, np.minimum(upper, obs) - above_left)
    return score


def discrete_crps(
    pmf: np.ndarray,
    observations: ArrayLike | float,
    grid: ArrayLike,
    *,
    reduction: Reduction = "none",
) -> np.ndarray | float:
    """Exact CRPS of a discrete common-grid distribution."""

    values = normalize_pmf(pmf)
    common_grid = _validate_grid(grid)
    obs = _broadcast_observations(observations, values.shape[:-1])
    score = _integrated_cdf_error(
        values,
        obs,
        common_grid,
        region_lower=float(common_grid[0]),
        region_upper=float(common_grid[-1]),
    )
    # When y is outside the forecast support, squared CDF error is one between
    # y and the nearest support endpoint.
    score += np.maximum(common_grid[0] - obs, 0.0)
    score += np.maximum(obs - common_grid[-1], 0.0)
    return _reduce(score, reduction)


def discrete_twcrps(
    pmf: np.ndarray,
    observations: ArrayLike | float,
    grid: ArrayLike,
    *,
    region_lower: float | None = None,
    region_upper: float | None = None,
    reduction: Reduction = "none",
) -> np.ndarray | float:
    """Threshold-weighted CRPS over ``[region_lower, region_upper]``.

    Set only ``region_upper`` for a lower-tail score and only ``region_lower``
    for an upper-tail score.  Missing bounds default to the grid endpoints.
    """

    score = _integrated_cdf_error(
        pmf,
        observations,
        grid,
        region_lower=region_lower,
        region_upper=region_upper,
    )
    return _reduce(score, reduction)


def discrete_lower_twcrps(
    pmf: np.ndarray,
    observations: ArrayLike | float,
    grid: ArrayLike,
    threshold: float,
    *,
    reduction: Reduction = "none",
) -> np.ndarray | float:
    """Convenience wrapper for CRPS integrated below ``threshold``."""

    return discrete_twcrps(
        pmf,
        observations,
        grid,
        region_upper=float(threshold),
        reduction=reduction,
    )


def discrete_upper_twcrps(
    pmf: np.ndarray,
    observations: ArrayLike | float,
    grid: ArrayLike,
    threshold: float,
    *,
    reduction: Reduction = "none",
) -> np.ndarray | float:
    """Convenience wrapper for CRPS integrated above ``threshold``."""

    return discrete_twcrps(
        pmf,
        observations,
        grid,
        region_lower=float(threshold),
        reduction=reduction,
    )


def observation_bin_indices(observations: ArrayLike | float, grid: ArrayLike) -> np.ndarray:
    """Map observations to nearest-grid Voronoi bins.

    Exact zero and exact cap observations map to the corresponding endpoint
    atoms.  Values outside support map to the nearest endpoint bin.
    """

    common_grid = _validate_grid(grid)
    obs = np.asarray(observations, dtype=float)
    if not np.isfinite(obs).all():
        raise ValueError("observations must be finite")
    edges = 0.5 * (common_grid[:-1] + common_grid[1:])
    return np.searchsorted(edges, obs, side="right")


def observation_bin_probability(
    pmf: np.ndarray,
    observations: ArrayLike | float,
    grid: ArrayLike,
) -> np.ndarray:
    """Return the PMF mass of the nearest grid bin for each observation."""

    values = normalize_pmf(pmf)
    common_grid = _validate_grid(grid)
    if values.shape[-1] != common_grid.size:
        raise ValueError("pmf grid axis and grid length do not match")
    obs = _broadcast_observations(observations, values.shape[:-1])
    indices = observation_bin_indices(obs, common_grid)
    return np.take_along_axis(values, indices[..., None], axis=-1)[..., 0]


def discrete_log_score(
    pmf: np.ndarray,
    observations: ArrayLike | float,
    grid: ArrayLike,
    *,
    min_probability: float = 1e-12,
    reduction: Reduction = "none",
) -> np.ndarray | float:
    """Negative log score based on observation-bin probability mass."""

    floor = float(min_probability)
    if not np.isfinite(floor) or floor <= 0.0 or floor > 1.0:
        raise ValueError("min_probability must be finite and lie in (0, 1]")
    probability = observation_bin_probability(pmf, observations, grid)
    score = -np.log(np.maximum(probability, floor))
    return _reduce(score, reduction)
