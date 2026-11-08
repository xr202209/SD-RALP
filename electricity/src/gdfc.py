"""Recursive generalised density forecast combination (GDFC).

Kapetanios, Mitchell, Price and Fawcett (2015).  For region ``s(y)``, the
combined probability mass function is

    p_t(y) = sum_i v[i, s(y)] q_it(y)
             / sum_{i,s} v[i, s] kappa_it[i, s],

where ``kappa_it[i, s]`` is component ``i``'s probability mass in region
``s``.  The coefficients lie on one global simplex and are re-estimated by
maximum log score every ``refit_days`` days, using only completed days.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from combination import CombinationResult, PanelData
from distributions import discrete_log_score, observation_bin_indices


_NEGATIVE_TOLERANCE = 1e-14
DEFAULT_MIN_PROBABILITY = 1e-12


@dataclass(frozen=True)
class KMPFFit:
    """Result of a multi-start KMPF coefficient fit."""

    coefficients: np.ndarray
    free_logits: np.ndarray
    objective: float
    success: bool
    message: str
    n_iter: int
    n_starts: int


@dataclass(frozen=True)
class _PreparedProblem:
    component_pmf: np.ndarray
    actual_bin: np.ndarray
    region_by_bin: np.ndarray
    actual_region: np.ndarray
    kappa: np.ndarray
    probability_at_actual: np.ndarray
    min_probability: float

    @property
    def n_models(self) -> int:
        return int(self.component_pmf.shape[1])

    @property
    def n_regions(self) -> int:
        return int(self.kappa.shape[2])


def _validate_structure(
    component_pmf: np.ndarray,
    grid: np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pmf = np.asarray(component_pmf, dtype=float)
    if pmf.ndim != 3:
        raise ValueError("component_pmf must have shape (n, k, g)")
    if min(pmf.shape) <= 0:
        raise ValueError("component_pmf dimensions must all be positive")
    if not np.isfinite(pmf).all():
        raise ValueError("component_pmf must be finite")
    if np.any(pmf < -_NEGATIVE_TOLERANCE):
        raise ValueError("component_pmf cannot contain negative mass")
    pmf = np.maximum(pmf, 0.0)
    component_totals = pmf.sum(axis=2, keepdims=True)
    if np.any(component_totals <= 0.0):
        raise ValueError("every component PMF must have positive total mass")
    # Renormalise to remove rounding error.
    pmf = pmf / component_totals

    grid_array = np.asarray(grid, dtype=float)
    if grid_array.ndim != 1 or grid_array.size != pmf.shape[2]:
        raise ValueError("grid must be one-dimensional with length g")
    if not np.isfinite(grid_array).all() or np.any(np.diff(grid_array) <= 0.0):
        raise ValueError("grid must be finite and strictly increasing")

    threshold_array = np.asarray(thresholds, dtype=float)
    if threshold_array.ndim != 1:
        raise ValueError("thresholds must be one-dimensional")
    if not np.isfinite(threshold_array).all():
        raise ValueError("thresholds must be finite")
    if threshold_array.size and np.any(np.diff(threshold_array) <= 0.0):
        raise ValueError("thresholds must be strictly increasing")

    # searchsorted(..., side="right") implements [r_{s-1}, r_s): a grid
    # point equal to a threshold belongs to the region on its right.
    region_by_bin = np.searchsorted(threshold_array, grid_array, side="right")
    n_regions = threshold_array.size + 1
    counts = np.bincount(region_by_bin, minlength=n_regions)
    if counts.size != n_regions or np.any(counts == 0):
        raise ValueError("thresholds must leave at least one grid bin in every region")
    return pmf, grid_array, threshold_array, region_by_bin.astype(np.int64)


def _kappa_from_regions(component_pmf: np.ndarray, region_by_bin: np.ndarray) -> np.ndarray:
    n, k, _ = component_pmf.shape
    n_regions = int(region_by_bin.max()) + 1
    kappa = np.empty((n, k, n_regions), dtype=float)
    for region in range(n_regions):
        kappa[:, :, region] = component_pmf[:, :, region_by_bin == region].sum(axis=2)
    return kappa


def global_simplex(
    free_logits: np.ndarray,
    n_models: int,
    n_regions: int,
) -> np.ndarray:
    """Map reference-category logits to one global ``(k, p)`` simplex.

    The final model-region cell is the reference category with logit zero.
    This removes the otherwise unidentified common shift in a full softmax.
    """

    if n_models <= 0 or n_regions <= 0:
        raise ValueError("n_models and n_regions must be positive")
    free = np.asarray(free_logits, dtype=float)
    expected = n_models * n_regions - 1
    if free.ndim != 1 or free.size != expected:
        raise ValueError(f"free_logits must have length {expected}")
    if not np.isfinite(free).all():
        raise ValueError("free_logits must be finite")

    full = np.concatenate([free, np.zeros(1, dtype=float)])
    full -= float(np.max(full))
    weights = np.exp(full)
    weights /= float(weights.sum())
    return weights.reshape(n_models, n_regions)


def _validate_actual_bin(actual_bin: np.ndarray, n: int, g: int) -> np.ndarray:
    raw = np.asarray(actual_bin)
    if raw.ndim != 1 or raw.size != n:
        raise ValueError("actual_bin must be one-dimensional with length n")
    if not np.isfinite(raw.astype(float)).all():
        raise ValueError("actual_bin must be finite")
    bins = raw.astype(np.int64)
    if not np.array_equal(raw, bins):
        raise ValueError("actual_bin must contain integer bin indices")
    if np.any((bins < 0) | (bins >= g)):
        raise ValueError("actual_bin contains an out-of-range bin index")
    return bins


def _prepare_problem(
    component_pmf: np.ndarray,
    actual_bin: np.ndarray,
    grid: np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> _PreparedProblem:
    pmf, _, _, region_by_bin = _validate_structure(component_pmf, grid, thresholds)
    n, k, g = pmf.shape
    bins = _validate_actual_bin(actual_bin, n, g)
    probability_at_actual = pmf[
        np.arange(n)[:, None],
        np.arange(k)[None, :],
        bins[:, None],
    ]
    floor = float(min_probability)
    if not np.isfinite(floor) or floor <= 0.0 or floor > 1.0:
        raise ValueError("min_probability must be finite and lie in (0, 1]")
    return _PreparedProblem(
        component_pmf=pmf,
        actual_bin=bins,
        region_by_bin=region_by_bin,
        actual_region=region_by_bin[bins],
        kappa=_kappa_from_regions(pmf, region_by_bin),
        probability_at_actual=probability_at_actual,
        min_probability=floor,
    )


def _objective_from_prepared(
    free_logits: np.ndarray,
    problem: _PreparedProblem,
) -> tuple[float, np.ndarray]:
    coefficients = global_simplex(
        free_logits,
        problem.n_models,
        problem.n_regions,
    )
    n = problem.component_pmf.shape[0]

    coefficients_at_actual = coefficients[:, problem.actual_region].T
    numerator = np.sum(problem.probability_at_actual * coefficients_at_actual, axis=1)
    denominator = np.einsum("nkp,kp->n", problem.kappa, coefficients, optimize=True)
    if np.any(denominator <= 0.0):
        raise FloatingPointError("KMPF likelihood has a non-positive normalizer")

    probability = numerator / denominator
    active = probability > problem.min_probability
    objective = -float(np.mean(np.log(np.maximum(probability, problem.min_probability))))

    # Observations below the probability floor contribute the constant
    # -log(floor) and have zero gradient.
    active_float = active.astype(float)

    region_indicator = np.eye(problem.n_regions, dtype=float)[problem.actual_region]
    numerator_term = np.einsum(
        "nk,nr,n->kr",
        problem.probability_at_actual,
        region_indicator,
        np.divide(
            active_float,
            numerator,
            out=np.zeros_like(numerator),
            where=numerator > 0.0,
        ),
        optimize=True,
    ) / n
    denominator_term = np.einsum(
        "nkp,n->kp",
        problem.kappa,
        active_float / denominator,
        optimize=True,
    ) / n
    gradient_coefficients = denominator_term - numerator_term

    # Softmax chain rule.  The final full logit is fixed at zero, so only the
    # first k*p-1 elements are returned to the optimiser.
    flat_coefficients = coefficients.ravel()
    flat_gradient = gradient_coefficients.ravel()
    centering = float(np.dot(flat_coefficients, flat_gradient))
    gradient_logits = flat_coefficients * (flat_gradient - centering)
    return objective, gradient_logits[:-1]


def fit_kmpf(
    component_pmf: np.ndarray,
    actual_bin: np.ndarray,
    grid: np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
    *,
    n_starts: int = 8,
    random_state: int | np.random.Generator | None = 0,
    maxiter: int = 1000,
    max_abs_logit: float = 30.0,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> KMPFFit:
    """Fit the coefficients by multi-start L-BFGS-B.

    The first start is the equal simplex and the others are random
    perturbations.  The logits are bounded by ``max_abs_logit``.
    """

    if n_starts < 1:
        raise ValueError("n_starts must be at least one")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if not np.isfinite(max_abs_logit) or max_abs_logit <= 0.0:
        raise ValueError("max_abs_logit must be finite and positive")

    problem = _prepare_problem(
        component_pmf,
        actual_bin,
        grid,
        thresholds,
        min_probability=min_probability,
    )
    n_free = problem.n_models * problem.n_regions - 1
    if n_free == 0:
        free = np.empty(0, dtype=float)
        objective, _ = _objective_from_prepared(free, problem)
        return KMPFFit(
            coefficients=np.ones((1, 1), dtype=float),
            free_logits=free,
            objective=objective,
            success=True,
            message="single coefficient; no optimisation required",
            n_iter=0,
            n_starts=1,
        )

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    starts = [np.zeros(n_free, dtype=float)]
    for start_index in range(1, n_starts):
        scale = (0.5, 1.0, 2.0)[(start_index - 1) % 3]
        starts.append(rng.normal(0.0, scale, size=n_free))

    results = []
    bounds = [(-max_abs_logit, max_abs_logit)] * n_free
    for start in starts:
        result = minimize(
            _objective_from_prepared,
            x0=start,
            args=(problem,),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "maxls": 50,
                "ftol": float(ftol),
                "gtol": float(gtol),
            },
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            results.append(result)

    if not results:
        raise RuntimeError("all KMPF optimisation starts failed")
    successful = [result for result in results if bool(result.success)]
    candidates = successful or results
    best = min(candidates, key=lambda result: float(result.fun))
    best_free = np.asarray(best.x, dtype=float)
    return KMPFFit(
        coefficients=global_simplex(best_free, problem.n_models, problem.n_regions),
        free_logits=best_free,
        objective=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
        n_iter=int(best.nit),
        n_starts=len(starts),
    )


def combine_pmf(
    component_pmf: np.ndarray,
    grid: np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Combine component PMFs using fixed KMPF region coefficients."""

    pmf, _, _, region_by_bin = _validate_structure(component_pmf, grid, thresholds)
    n, k, _ = pmf.shape
    n_regions = int(region_by_bin.max()) + 1
    raw_coefficients = np.asarray(coefficients, dtype=float)
    if raw_coefficients.shape != (k, n_regions):
        raise ValueError(f"coefficients must have shape {(k, n_regions)}")
    if not np.isfinite(raw_coefficients).all():
        raise ValueError("coefficients must be finite")
    if np.any(raw_coefficients < -_NEGATIVE_TOLERANCE):
        raise ValueError("coefficients must be non-negative")
    raw_coefficients = np.maximum(raw_coefficients, 0.0)
    total = float(raw_coefficients.sum())
    if total <= 0.0:
        raise ValueError("at least one coefficient must be positive")
    # The combined PMF does not depend on the scale of the coefficients.
    identified = raw_coefficients / total

    coefficient_by_bin = identified[:, region_by_bin]
    unnormalised = np.einsum("nkg,kg->ng", pmf, coefficient_by_bin, optimize=True)
    kappa = _kappa_from_regions(pmf, region_by_bin)
    denominator = np.einsum("nkp,kp->n", kappa, identified, optimize=True)
    if np.any(denominator <= 0.0):
        raise ValueError("coefficients assign zero mass to an observation's component support")
    combined = unnormalised / denominator[:, None]
    # Remove rounding error.
    combined /= combined.sum(axis=1, keepdims=True)
    if combined.shape != (n, pmf.shape[2]) or not np.isfinite(combined).all():
        raise FloatingPointError("invalid combined PMF")
    return combined


def kmpf_thresholds(history_actual: np.ndarray, p: int, grid: np.ndarray) -> np.ndarray:
    """Expanding-history equal-probability sieve with strictly ordered grid edges."""

    if p < 2 or p >= len(grid):
        raise ValueError("KMPF region count is incompatible with the grid")
    desired = np.quantile(np.asarray(history_actual, dtype=float).reshape(-1), np.arange(1, p) / p)
    edges = 0.5 * (grid[:-1] + grid[1:])
    indices = np.searchsorted(edges, desired, side="left")
    indices = np.clip(indices, 0, len(edges) - 1).astype(int)
    for position in range(1, len(indices)):
        indices[position] = max(indices[position], indices[position - 1] + 1)
    if indices[-1] >= len(edges):
        indices[-1] = len(edges) - 1
        for position in range(len(indices) - 2, -1, -1):
            indices[position] = min(indices[position], indices[position + 1] - 1)
    if np.any(np.diff(indices) <= 0):
        raise RuntimeError("unable to construct strictly increasing KMPF thresholds")
    return edges[indices]


def recursive_kmpf(
    initial: PanelData,
    score: PanelData,
    *,
    p: int,
    refit_days: int,
    n_starts: int,
    max_iter: int,
    random_seed: int,
    collect_pmf: bool,
    min_probability: float = 1e-12,
    evaluate_log_score: bool = True,
) -> tuple[float, np.ndarray | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Recursively refit KMPF using only days completed before each block."""

    all_pmf = np.concatenate([initial.component_pmf, score.component_pmf], axis=0)
    all_actual = np.concatenate([initial.actual, score.actual], axis=0)
    initial_days = initial.days
    output = (
        np.empty((score.days, score.slots, score.grid_size), dtype=np.float32)
        if collect_pmf
        else None
    )
    log_score_sum = 0.0
    observation_count = 0
    fit_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for block_start in range(0, score.days, int(refit_days)):
        block_end = min(score.days, block_start + int(refit_days))
        history_end = initial_days + block_start
        history_pmf = all_pmf[:history_end].reshape(-1, initial.model_count, initial.grid_size)
        history_actual = all_actual[:history_end].reshape(-1)
        history_bins = observation_bin_indices(history_actual, initial.grid).reshape(-1)
        thresholds = kmpf_thresholds(history_actual, int(p), initial.grid)
        fit = fit_kmpf(
            history_pmf,
            history_bins,
            initial.grid,
            thresholds,
            n_starts=int(n_starts),
            random_state=int(random_seed) + 1009 * int(p) + block_start,
            maxiter=int(max_iter),
            min_probability=float(min_probability),
        )
        if not fit.success:
            raise RuntimeError(
                f"KMPF refit did not converge for p={p}, block_start={block_start}: "
                f"{fit.message}"
            )
        target_pmf = all_pmf[initial_days + block_start : initial_days + block_end]
        flat_target = target_pmf.reshape(-1, initial.model_count, initial.grid_size)
        combined_flat = combine_pmf(
            flat_target,
            initial.grid,
            thresholds,
            fit.coefficients,
        )
        combined = combined_flat.reshape(block_end - block_start, score.slots, score.grid_size)
        if evaluate_log_score:
            target_actual = score.actual[block_start:block_end]
            block_log_score = np.asarray(
                discrete_log_score(
                    combined,
                    target_actual,
                    initial.grid,
                    min_probability=float(min_probability),
                ),
                dtype=float,
            )
            log_score_sum += float(block_log_score.sum())
            observation_count += int(block_log_score.size)
        if output is not None:
            output[block_start:block_end] = combined.astype(np.float32)

        fit_rows.append(
            {
                "block_start_day_index": int(block_start),
                "block_end_day_index": int(block_end - 1),
                "history_days": int(history_end),
                "p": int(p),
                "thresholds": json.dumps(thresholds.tolist()),
                "fit_objective": float(fit.objective),
                "fit_success": bool(fit.success),
                "fit_message": fit.message,
                "fit_iterations": int(fit.n_iter),
                "n_starts": int(fit.n_starts),
            }
        )
        for day_index in range(block_start, block_end):
            for model_index in range(initial.model_count):
                for region_index in range(int(p)):
                    coefficient_rows.append(
                        {
                            "day_index": int(day_index),
                            "region": f"R{region_index + 1}",
                            "model_index": int(model_index),
                            "weight": float(fit.coefficients[model_index, region_index]),
                            "coefficient_type": "global_simplex_v",
                        }
                    )
    mean_log_score = (
        log_score_sum / observation_count if observation_count else float("nan")
    )
    return mean_log_score, output, fit_rows, coefficient_rows


def wrap_kmpf_result(panel: PanelData, pmf: np.ndarray) -> CombinationResult:
    values = np.asarray(pmf, dtype=float)
    values /= values.sum(axis=-1, keepdims=True)
    dummy = np.repeat(1.0 / panel.model_count, panel.model_count)
    trace = np.broadcast_to(dummy[None, None, :], (panel.days, 3, panel.model_count)).copy()
    result = CombinationResult(
        method="Recursive GDFC",
        pmf=values,
        cdf=np.cumsum(values, axis=-1),
        weights_trace=trace,
        normalizer=np.ones((panel.days, panel.slots)),
        diagnostics={"weights_trace_is_dummy": True},
    )
    result.validate(panel)
    return result
