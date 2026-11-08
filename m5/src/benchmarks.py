"""Benchmark combinations of the M5 application.

TVW and DTVW follow the latent-state and diversity recursions of Luo, Kang and
Luo (2025), with the likelihood of a linear density pool.  At each origin the
log likelihood is averaged across series, so that one weekly panel does not
dominate the particle weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby
from typing import Any

import numpy as np
from scipy.optimize import minimize

import regional_pool as base
import quantile_average


EPS = 1e-12
GDFC_REGIONS = ("L", "C", "U")


def normalized_pdf_cdf(grid: np.ndarray, pdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density = np.maximum(np.asarray(pdf, dtype=float), 0.0)
    density /= max(base.trapz(density, grid), EPS)
    return density, base.cumulative_from_pdf(grid, density)


def linear_pool(rec: base.Record, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return normalized_pdf_cdf(rec.grid, np.asarray(weights, dtype=float) @ rec.comp_pdfs)


def quantile_average_pool(rec: base.Record) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-average pool of the component distributions."""
    dist = quantile_average.distribution(rec)
    return dist.pdf, dist.cdf


def component_density_at_y(rec: base.Record) -> np.ndarray:
    return np.asarray(
        [base.density_at_value(rec.grid, pdf, rec.y) for pdf in rec.comp_pdfs],
        dtype=float,
    )


def day_groups(records: list[base.Record]) -> list[tuple[int, list[base.Record]]]:
    ordered = sorted(records, key=lambda item: (item.day, item.series_id))
    return [(int(day), list(group)) for day, group in groupby(ordered, key=lambda item: item.day)]


def daily_component_log_scores(records: list[base.Record]) -> list[tuple[int, np.ndarray]]:
    return [
        (
            day,
            np.mean(np.log(np.vstack([component_density_at_y(rec) for rec in group])), axis=0),
        )
        for day, group in day_groups(records)
    ]


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    exp_values = np.exp(shifted)
    return exp_values / max(float(exp_values.sum()), EPS)


def row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), EPS)


def fit_log_score_pool(records: list[base.Record]) -> tuple[np.ndarray, str]:
    density = np.vstack([component_density_at_y(rec) for rec in records])
    k = density.shape[1]

    def objective(weights: np.ndarray) -> float:
        return float(-np.mean(np.log(np.maximum(density @ weights, EPS))))

    result = minimize(
        objective,
        np.repeat(1.0 / k, k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Log-score pool optimization failed: {result.message}")
    weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    weights /= max(float(weights.sum()), EPS)
    return weights, str(result.message)


def fit_bma_weights(records: list[base.Record]) -> np.ndarray:
    daily = daily_component_log_scores(records)
    return stable_softmax(np.sum([scores for _, scores in daily], axis=0))


@dataclass(frozen=True)
class GDFCFit:
    """Recursive three-region sieve fit adapted from Kapetanios et al. (2015)."""

    coefficients: np.ndarray
    logits: np.ndarray
    objective: float
    success: bool
    message: str
    iterations: int
    gradient_norm: float
    n_records: int


def gdfc_record_terms(rec: base.Record) -> tuple[np.ndarray, np.ndarray, int]:
    """Return component region masses, realised densities, and realised region."""
    cached = getattr(rec, "_gdfc_terms_cache", None)
    if cached is not None:
        return cached
    cdf_low = np.asarray(
        [base.cdf_at_value(rec.grid, cdf, rec.c_low) for cdf in rec.comp_cdfs],
        dtype=float,
    )
    cdf_high = np.asarray(
        [base.cdf_at_value(rec.grid, cdf, rec.c_high) for cdf in rec.comp_cdfs],
        dtype=float,
    )
    masses = np.column_stack((cdf_low, cdf_high - cdf_low, 1.0 - cdf_high))
    masses = np.maximum(masses, 0.0)
    masses /= np.maximum(masses.sum(axis=1, keepdims=True), EPS)
    density = np.maximum(component_density_at_y(rec), EPS)
    realised_region = 0 if rec.y <= rec.c_low else 1 if rec.y <= rec.c_high else 2
    cached = (masses, density, realised_region)
    setattr(rec, "_gdfc_terms_cache", cached)
    return cached


def stack_gdfc_terms(records: list[base.Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("GDFC estimation requires at least one realised forecast record")
    terms = [gdfc_record_terms(rec) for rec in records]
    masses = np.stack([term[0] for term in terms])
    density = np.stack([term[1] for term in terms])
    realised_region = np.asarray([term[2] for term in terms], dtype=int)
    return masses, density, realised_region


def gdfc_objective_gradient(
    logits: np.ndarray,
    masses: np.ndarray,
    density: np.ndarray,
    realised_region: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Negative log score and analytic gradient for the generalized pool."""
    k = int(density.shape[1])
    coefficients_flat = stable_softmax(np.asarray(logits, dtype=float))
    coefficients = coefficients_flat.reshape(k, len(GDFC_REGIONS))
    normalizer = np.maximum(np.einsum("nkj,kj->n", masses, coefficients), EPS)
    selected_coefficients = coefficients[:, realised_region].T
    numerator = np.maximum(np.sum(density * selected_coefficients, axis=1), EPS)
    objective = float(np.mean(np.log(normalizer) - np.log(numerator)))

    direct_gradient = np.mean(masses / normalizer[:, None, None], axis=0)
    n_records = max(int(len(realised_region)), 1)
    for region_idx in range(len(GDFC_REGIONS)):
        mask = realised_region == region_idx
        if np.any(mask):
            direct_gradient[:, region_idx] -= np.sum(
                density[mask] / numerator[mask, None], axis=0
            ) / n_records
    direct_flat = direct_gradient.ravel()
    gradient = coefficients_flat * (
        direct_flat - float(np.dot(coefficients_flat, direct_flat))
    )
    return objective, gradient


def fit_gdfc_coefficients(
    records: list[base.Record],
    initial_logits: np.ndarray | None = None,
    maxiter: int = 250,
) -> GDFCFit:
    """Estimate positive regional coefficients by recursive log-score minimization."""
    masses, density, realised_region = stack_gdfc_terms(records)
    n_parameters = int(density.shape[1] * len(GDFC_REGIONS))
    start = np.zeros(n_parameters, dtype=float) if initial_logits is None else np.asarray(initial_logits, dtype=float).copy()
    if start.shape != (n_parameters,) or not np.isfinite(start).all():
        raise ValueError("Invalid GDFC initial logits")
    start_objective, _ = gdfc_objective_gradient(start, masses, density, realised_region)
    result = minimize(
        lambda value: gdfc_objective_gradient(value, masses, density, realised_region),
        start,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-8, "maxls": 40},
    )
    candidate = np.asarray(result.x, dtype=float)
    objective, gradient = gdfc_objective_gradient(candidate, masses, density, realised_region)
    if not np.isfinite(candidate).all() or not math.isfinite(objective) or objective > start_objective + 1e-8:
        candidate = start
        objective, gradient = gdfc_objective_gradient(candidate, masses, density, realised_region)
        success = False
        message = f"optimizer fallback to warm start: {result.message}"
    else:
        success = bool(result.success)
        message = str(result.message)
    coefficients = stable_softmax(candidate).reshape(density.shape[1], len(GDFC_REGIONS))
    return GDFCFit(
        coefficients=coefficients,
        logits=candidate,
        objective=float(objective),
        success=success,
        message=message,
        iterations=int(getattr(result, "nit", 0)),
        gradient_norm=float(np.linalg.norm(gradient)),
        n_records=int(len(records)),
    )


def gdfc_pool(rec: base.Record, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the normalized piecewise generalized density combination."""
    coeff = np.asarray(coefficients, dtype=float)
    expected_shape = (rec.comp_pdfs.shape[0], len(GDFC_REGIONS))
    if coeff.shape != expected_shape or not np.isfinite(coeff).all() or np.any(coeff < 0.0):
        raise ValueError(f"Invalid GDFC coefficients; expected shape {expected_shape}")
    raw_pdf = np.zeros_like(rec.grid, dtype=float)
    for region_idx, region in enumerate(GDFC_REGIONS):
        mask = base.regional_mask(rec, region)
        raw_pdf[mask] = coeff[:, region_idx] @ rec.comp_pdfs[:, mask]
    return normalized_pdf_cdf(rec.grid, raw_pdf)


@dataclass(frozen=True)
class ParticleSpec:
    particles: int
    state_noise: float
    coefficient_noise: float
    resample_ess_fraction: float
    alpha0: float
    alpha1: float
    alpha2: float
    random_seed: int


@dataclass
class ParticleState:
    x: np.ndarray
    alpha: np.ndarray
    log_weights: np.ndarray
    rng: np.random.Generator


def particle_spec(config: dict[str, Any], alpha1: float, alpha2: float) -> ParticleSpec:
    pf = config["benchmarks"]["particle_filter"]
    return ParticleSpec(
        particles=int(pf["particles"]),
        state_noise=float(pf["state_noise"]),
        coefficient_noise=float(pf["coefficient_noise"]),
        resample_ess_fraction=float(pf["resample_ess_fraction"]),
        alpha0=float(pf["alpha0"]),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
        random_seed=int(pf["random_seed"]),
    )


def initial_particle_state(spec: ParticleSpec, model_count: int) -> ParticleState:
    return ParticleState(
        x=np.zeros((spec.particles, model_count), dtype=float),
        alpha=np.tile(np.asarray([spec.alpha0, spec.alpha1, spec.alpha2]), (spec.particles, 1)),
        log_weights=np.repeat(-np.log(spec.particles), spec.particles),
        rng=np.random.default_rng(spec.random_seed),
    )


def density_means(rec: base.Record) -> np.ndarray:
    return np.asarray([base.trapz(rec.grid * pdf, rec.grid) for pdf in rec.comp_pdfs])


def panel_diversity(group: list[base.Record]) -> np.ndarray:
    """Average the scaled DTVW diversity vector over the M5 panel at one origin."""

    vectors = []
    for rec in group:
        predictions = density_means(rec)
        pairwise = predictions[:, None] - predictions[None, :]
        denominator = float(np.sum(pairwise * pairwise))
        if denominator <= EPS:
            vectors.append(np.zeros_like(predictions))
            continue
        vectors.append(np.sum((predictions[:, None] - predictions[None, :]) ** 2, axis=1) / denominator)
    return np.mean(vectors, axis=0)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def resample_pending_payload(payload: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "density": payload["density"],
        "component_weights": payload["component_weights"][indices],
    }


def maybe_resample_particles(
    state: ParticleState,
    spec: ParticleSpec,
    pending: list[tuple[int, dict[str, np.ndarray]]],
    available: list[tuple[int, dict[str, np.ndarray]]],
) -> tuple[list[tuple[int, dict[str, np.ndarray]]], list[tuple[int, dict[str, np.ndarray]]]]:
    particle_weights = stable_softmax(state.log_weights)
    ess_fraction = 1.0 / (spec.particles * float(np.sum(particle_weights**2)))
    if ess_fraction >= spec.resample_ess_fraction:
        return pending, available
    indices = systematic_resample(particle_weights, state.rng)
    state.x = state.x[indices]
    state.alpha = state.alpha[indices]
    state.log_weights = np.repeat(-np.log(spec.particles), spec.particles)
    pending = [(day, resample_pending_payload(payload, indices)) for day, payload in pending]
    available = [(day, resample_pending_payload(payload, indices)) for day, payload in available]
    return pending, available


def apply_delayed_particle_likelihood(
    state: ParticleState,
    spec: ParticleSpec,
    payload: dict[str, np.ndarray],
) -> None:
    panel_log_likelihood = np.mean(
        np.log(np.maximum(payload["density"] @ payload["component_weights"].T, EPS)),
        axis=0,
    )
    state.log_weights += panel_log_likelihood
    state.log_weights -= float(np.max(state.log_weights))
