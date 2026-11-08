"""Risk-averse battery scheduling for the electricity application.

For each delivery day a common empirical copula is formed from the 180 previous
realised daily price curves, each rank path replicated ten times on a stratified
within-rank grid (1,800 scenarios).  Every method maps the same uniforms through
its 96 predictive marginals, and a one-stage mean-CVaR schedule is solved as a
linear program.  The decision loss is the realised cost of the schedule minus
the cost of the perfect-information schedule.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class BatterySpec:
    """Battery, risk and scenario settings."""

    power_mw: float = 1.0
    energy_mwh: float = 2.0
    interval_hours: float = 0.25
    round_trip_efficiency: float = 0.90
    initial_soc_fraction: float = 0.50
    degradation_cny_per_mwh: float = 20.0
    max_efc_per_day: float = 2.0
    risk_weight: float = 0.50
    cvar_alpha: float = 0.95
    copula_history_days: int = 180
    copula_replications: int = 10
    copula_random_seed: int = 20260720
    perfect_information_tolerance_cny: float = 1e-5

    @property
    def charge_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def discharge_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc_fraction * self.energy_mwh

    @property
    def scenario_count(self) -> int:
        return self.copula_history_days * self.copula_replications

    def validate(self) -> None:
        positive = {
            "power_mw": self.power_mw,
            "energy_mwh": self.energy_mwh,
            "interval_hours": self.interval_hours,
            "round_trip_efficiency": self.round_trip_efficiency,
            "copula_history_days": self.copula_history_days,
            "copula_replications": self.copula_replications,
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError(f"battery/scenario values must be positive: {positive}")
        if self.round_trip_efficiency > 1.0:
            raise ValueError("round-trip efficiency cannot exceed one")
        if not 0.0 <= self.initial_soc_fraction <= 1.0:
            raise ValueError("initial SOC fraction must lie in [0, 1]")
        if self.degradation_cny_per_mwh < 0.0:
            raise ValueError("degradation cost cannot be negative")
        if self.max_efc_per_day <= 0.0:
            raise ValueError("max EFC per day must be positive")
        if not 0.0 <= self.risk_weight <= 1.0:
            raise ValueError("risk weight must lie in [0, 1]")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("CVaR alpha must lie in (0, 1)")
        if self.perfect_information_tolerance_cny < 0.0:
            raise ValueError("perfect-information tolerance cannot be negative")


@dataclass(frozen=True)
class BatteryMatrices:
    slots: int
    base_variables: int
    charge_slice: slice
    discharge_slice: slice
    energy_slice: slice
    power_constraints: sparse.csr_matrix
    power_rhs: np.ndarray
    energy_equalities: sparse.csr_matrix
    energy_rhs: np.ndarray
    base_bounds: tuple[tuple[float | None, float | None], ...]


def build_battery_matrices(spec: BatterySpec, slots: int = 96) -> BatteryMatrices:
    """Create method/day-invariant battery feasibility matrices."""

    charge = slice(0, slots)
    discharge = slice(slots, 2 * slots)
    energy = slice(2 * slots, 3 * slots)
    base_variables = 3 * slots

    power = sparse.lil_matrix((slots, base_variables), dtype=float)
    for slot in range(slots):
        power[slot, charge.start + slot] = 1.0
        power[slot, discharge.start + slot] = 1.0
    power_rhs = np.full(slots, spec.power_mw, dtype=float)

    equalities = sparse.lil_matrix((slots + 1, base_variables), dtype=float)
    energy_rhs = np.zeros(slots + 1, dtype=float)
    dt = spec.interval_hours
    eta_c = spec.charge_efficiency
    eta_d = spec.discharge_efficiency
    for slot in range(slots):
        equalities[slot, energy.start + slot] = 1.0
        equalities[slot, charge.start + slot] = -eta_c * dt
        equalities[slot, discharge.start + slot] = dt / eta_d
        if slot == 0:
            energy_rhs[slot] = spec.initial_energy_mwh
        else:
            equalities[slot, energy.start + slot - 1] = -1.0
    equalities[slots, energy.stop - 1] = 1.0
    energy_rhs[slots] = spec.initial_energy_mwh

    bounds = tuple(
        [(0.0, spec.power_mw)] * (2 * slots)
        + [(0.0, spec.energy_mwh)] * slots
    )
    return BatteryMatrices(
        slots=slots,
        base_variables=base_variables,
        charge_slice=charge,
        discharge_slice=discharge,
        energy_slice=energy,
        power_constraints=power.tocsr(),
        power_rhs=power_rhs,
        energy_equalities=equalities.tocsr(),
        energy_rhs=energy_rhs,
        base_bounds=bounds,
    )


def _stable_uint64(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def build_checkerboard_copulas(
    actual_curves: pd.DataFrame,
    forecast_days: np.ndarray,
    history_days: int,
    replications: int,
    random_seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Build common stratified checkerboard uniforms from strict-prior curves.

    Ties are ordered by a deterministic per-cell random number.  Every one of
    the N empirical rank paths is replicated K times, and the K copies occupy
    all K equally spaced subcells within each marginal rank cell.  Therefore
    each slot contains exactly the S=N*K midpoint-stratified uniforms while
    retaining the historical coarse rank-path dependence.
    """

    if history_days < 2 or replications < 1:
        raise ValueError("checkerboard history and replications are invalid")
    rows, slots = actual_curves.shape
    if slots != 96:
        raise RuntimeError(f"expected 96 actual-price slots, got {slots}")

    tie_jitter = np.empty((rows, slots), dtype=float)
    subcell_permutations = np.empty((rows, slots, replications), dtype=np.int16)
    for row, historical_day in enumerate(actual_curves.index):
        day_key = str(historical_day.date())
        for slot in range(slots):
            cell_rng = np.random.default_rng(
                _stable_uint64(random_seed, "checkerboard", day_key, slot + 1)
            )
            tie_jitter[row, slot] = cell_rng.random()
            subcell_permutations[row, slot] = cell_rng.permutation(replications)
    tie_jitter_frame = pd.DataFrame(
        tie_jitter, index=actual_curves.index, columns=actual_curves.columns
    )
    row_positions = {day: pos for pos, day in enumerate(actual_curves.index)}
    expected_uniforms = (np.arange(history_days * replications) + 0.5) / (
        history_days * replications
    )

    copulas: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    for raw_day in forecast_days:
        day = pd.Timestamp(raw_day)
        history = actual_curves.loc[actual_curves.index < day].tail(history_days)
        if len(history) != history_days:
            raise RuntimeError(
                f"{day.date()}: expected exactly {history_days} strict-prior curves, "
                f"got {len(history)}"
            )
        if not bool(history.index.max() < day):
            raise RuntimeError(f"{day.date()}: copula history is not strictly prior")

        values = history.to_numpy(dtype=float)
        history_rows = np.asarray([row_positions[item] for item in history.index], dtype=int)
        jitter = tie_jitter_frame.loc[history.index].to_numpy(dtype=float)
        zero_based_ranks = np.empty((history_days, slots), dtype=np.int32)
        tied_cells = 0
        for slot in range(slots):
            order = np.lexsort((jitter[:, slot], values[:, slot]))
            zero_based_ranks[order, slot] = np.arange(history_days, dtype=np.int32)
            _, inverse, counts = np.unique(
                values[:, slot], return_inverse=True, return_counts=True
            )
            tied_cells += int(np.sum(counts[inverse] > 1))

        uniforms = np.empty((history_days * replications, slots), dtype=float)
        subcell_midpoints = (np.arange(replications, dtype=float) + 0.5) / replications
        for local_row, global_row in enumerate(history_rows):
            target = slice(local_row * replications, (local_row + 1) * replications)
            for slot in range(slots):
                permutation = subcell_permutations[global_row, slot]
                uniforms[target, slot] = (
                    zero_based_ranks[local_row, slot]
                    + subcell_midpoints[permutation]
                ) / history_days

        if not np.all((uniforms > 0.0) & (uniforms < 1.0)):
            raise RuntimeError(f"{day.date()}: checkerboard uniforms left (0,1)")
        stratification_error = float(
            max(
                np.max(np.abs(np.sort(uniforms[:, slot]) - expected_uniforms))
                for slot in range(slots)
            )
        )
        if stratification_error > 1e-12:
            raise RuntimeError(
                f"{day.date()}: marginal checkerboard stratification error "
                f"{stratification_error:.3g}"
            )

        uniforms = np.ascontiguousarray(uniforms, dtype=np.float64)
        copulas.append(uniforms)
        audit.append(
            {
                "delivery_day": str(day.date()),
                "history_first_day": str(history.index.min().date()),
                "history_last_day": str(history.index.max().date()),
                "history_days": int(history_days),
                "replications_per_rank_path": int(replications),
                "scenario_count": int(len(uniforms)),
                "strictly_prior": True,
                "tied_historical_cells": int(tied_cells),
                "fixed_seed_tie_breaking": True,
                "marginal_midpoint_stratification_max_error": stratification_error,
                "minimum_uniform": float(uniforms.min()),
                "maximum_uniform": float(uniforms.max()),
                "uniform_sha256": hashlib.sha256(uniforms.tobytes()).hexdigest(),
            }
        )
    return copulas, audit


def pmf_to_scenarios(
    pmf_day: np.ndarray,
    grid: np.ndarray,
    common_uniforms: np.ndarray,
) -> np.ndarray:
    if pmf_day.shape != (96, len(grid)):
        raise ValueError(f"malformed daily PMF shape {pmf_day.shape}")
    if common_uniforms.ndim != 2 or common_uniforms.shape[1] != 96:
        raise ValueError("common checkerboard uniforms must have shape S x 96")
    cdf = np.maximum.accumulate(np.cumsum(pmf_day, axis=-1), axis=-1)
    cdf[:, -1] = 1.0
    scenarios = np.empty_like(common_uniforms, dtype=float)
    for slot in range(96):
        indices = np.searchsorted(cdf[slot], common_uniforms[:, slot], side="left")
        scenarios[:, slot] = grid[np.minimum(indices, len(grid) - 1)]
    if not np.isfinite(scenarios).all():
        raise RuntimeError("non-finite prices generated from full PMFs")
    if np.any(scenarios < grid[0]) or np.any(scenarios > grid[-1]):
        raise RuntimeError("generated scenario left the price support")
    return scenarios


def scenario_cost_coefficients(
    prices: np.ndarray,
    spec: BatterySpec,
) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or values.shape[1] != 96 or not np.isfinite(values).all():
        raise ValueError("price scenarios must be a finite S x 96 array")
    dt = spec.interval_hours
    # Buying costs the price, selling earns it; wear is charged per MWh discharged.
    charge_coefficients = dt * values
    discharge_coefficients = dt * (-values + spec.degradation_cny_per_mwh)
    return np.concatenate(
        [charge_coefficients, discharge_coefficients, np.zeros_like(values)], axis=1
    )


def cell_side_equivalent_full_cycles(
    charge: np.ndarray,
    discharge: np.ndarray,
    spec: BatterySpec,
) -> float:
    """Cell-side energy throughput divided by twice nameplate energy."""

    cell_throughput_mwh = spec.interval_hours * np.sum(
        spec.charge_efficiency * np.asarray(charge, dtype=float)
        + np.asarray(discharge, dtype=float) / spec.discharge_efficiency
    )
    return float(cell_throughput_mwh / (2.0 * spec.energy_mwh))


def operating_inequalities(
    spec: BatterySpec,
    matrices: BatteryMatrices,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Shared-inverter inequalities plus the daily cycle cap."""

    cycle_row = sparse.lil_matrix((1, matrices.base_variables), dtype=float)
    charge_coefficient = (
        spec.interval_hours
        * spec.charge_efficiency
        / (2.0 * spec.energy_mwh)
    )
    discharge_coefficient = (
        spec.interval_hours
        / spec.discharge_efficiency
        / (2.0 * spec.energy_mwh)
    )
    cycle_row[0, matrices.charge_slice] = charge_coefficient
    cycle_row[0, matrices.discharge_slice] = discharge_coefficient
    constraints = sparse.vstack(
        [matrices.power_constraints, cycle_row.tocsr()], format="csr"
    )
    rhs = np.concatenate(
        [matrices.power_rhs, np.asarray([spec.max_efc_per_day], dtype=float)]
    )
    return constraints, rhs


def cycle_cap_metrics(
    charge: np.ndarray,
    discharge: np.ndarray,
    spec: BatterySpec,
) -> dict[str, float]:
    cell_efc = cell_side_equivalent_full_cycles(charge, discharge, spec)
    residual = cell_efc - spec.max_efc_per_day
    return {
        "cell_side_equivalent_full_cycles": cell_efc,
        "cell_side_efc_cap_slack": float(-residual),
        "cell_side_efc_cap_violation": float(max(0.0, residual)),
    }


def solve_risk_averse_schedule(
    scenarios: np.ndarray,
    spec: BatterySpec,
    matrices: BatteryMatrices,
) -> dict[str, Any]:
    """Solve the one-stage (1-lambda) mean + lambda CVaR battery LP."""

    coefficients = scenario_cost_coefficients(scenarios, spec)
    operating_constraints, operating_rhs = operating_inequalities(spec, matrices)
    scenario_count = len(coefficients)
    if scenario_count != spec.scenario_count:
        raise RuntimeError(
            f"expected {spec.scenario_count} scenarios, got {scenario_count}"
        )
    base_n = matrices.base_variables
    eta_index = base_n
    z_start = base_n + 1
    variable_count = z_start + scenario_count

    objective = np.zeros(variable_count, dtype=float)
    objective[:base_n] = (1.0 - spec.risk_weight) * coefficients.mean(axis=0)
    objective[eta_index] = spec.risk_weight
    objective[z_start:] = spec.risk_weight / (
        (1.0 - spec.cvar_alpha) * scenario_count
    )

    cvar_rows = sparse.hstack(
        [
            sparse.csr_matrix(coefficients),
            -sparse.csr_matrix(np.ones((scenario_count, 1), dtype=float)),
            -sparse.identity(scenario_count, format="csr", dtype=float),
        ],
        format="csr",
    )
    power_padded = sparse.hstack(
        [
            operating_constraints,
            sparse.csr_matrix((operating_constraints.shape[0], 1 + scenario_count)),
        ],
        format="csr",
    )
    a_ub = sparse.vstack([power_padded, cvar_rows], format="csr")
    b_ub = np.concatenate([operating_rhs, np.zeros(scenario_count)])
    a_eq = sparse.hstack(
        [
            matrices.energy_equalities,
            sparse.csr_matrix((matrices.slots + 1, 1 + scenario_count)),
        ],
        format="csr",
    )
    bounds = matrices.base_bounds + ((None, None),) + ((0.0, None),) * scenario_count
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=matrices.energy_rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(
            f"risk-averse battery LP failed: {result.status} {result.message}"
        )

    decision = result.x[:base_n]
    losses = coefficients @ decision
    eta = float(result.x[eta_index])
    expected = float(losses.mean())
    cvar = float(
        eta + np.maximum(losses - eta, 0.0).mean() / (1.0 - spec.cvar_alpha)
    )
    equality_residual = matrices.energy_equalities @ decision - matrices.energy_rhs
    inverter_residual = matrices.power_constraints @ decision - matrices.power_rhs
    return {
        "charge": decision[matrices.charge_slice].copy(),
        "discharge": decision[matrices.discharge_slice].copy(),
        "energy": decision[matrices.energy_slice].copy(),
        "forecast_expected_cost": expected,
        "forecast_var_threshold": eta,
        "forecast_cvar_cost": cvar,
        "forecast_objective": (1.0 - spec.risk_weight) * expected
        + spec.risk_weight * cvar,
        "solver_iterations": int(result.nit),
        "maximum_energy_equality_residual": float(
            np.max(np.abs(equality_residual))
        ),
        "maximum_shared_inverter_violation": float(
            max(0.0, np.max(inverter_residual))
        ),
    }


def solve_perfect_information(
    realised_prices: np.ndarray,
    spec: BatterySpec,
    matrices: BatteryMatrices,
) -> dict[str, Any]:
    coefficients = scenario_cost_coefficients(
        np.asarray(realised_prices, dtype=float).reshape(1, 96), spec
    )[0]
    operating_constraints, operating_rhs = operating_inequalities(spec, matrices)
    result = linprog(
        coefficients,
        A_ub=operating_constraints,
        b_ub=operating_rhs,
        A_eq=matrices.energy_equalities,
        b_eq=matrices.energy_rhs,
        bounds=matrices.base_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(
            f"perfect-information LP failed: {result.status} {result.message}"
        )
    decision = result.x
    charge = decision[matrices.charge_slice]
    discharge = decision[matrices.discharge_slice]
    pi_cycle_metrics = cycle_cap_metrics(charge, discharge, spec)
    return {
        "perfect_information_cost": float(coefficients @ decision),
        "perfect_information_charge_mwh": float(spec.interval_hours * charge.sum()),
        "perfect_information_discharge_mwh": float(
            spec.interval_hours * discharge.sum()
        ),
        "perfect_information_max_simultaneous_charge_discharge_mw": float(
            np.minimum(charge, discharge).max()
        ),
        "perfect_information_maximum_energy_equality_residual": float(
            np.max(
                np.abs(matrices.energy_equalities @ decision - matrices.energy_rhs)
            )
        ),
        "perfect_information_maximum_shared_inverter_violation": float(
            max(
                0.0,
                np.max(
                    matrices.power_constraints @ decision - matrices.power_rhs
                ),
            )
        ),
        "perfect_information_cell_side_equivalent_full_cycles": pi_cycle_metrics[
            "cell_side_equivalent_full_cycles"
        ],
        "perfect_information_cell_side_efc_cap_slack": pi_cycle_metrics[
            "cell_side_efc_cap_slack"
        ],
        "perfect_information_cell_side_efc_cap_violation": pi_cycle_metrics[
            "cell_side_efc_cap_violation"
        ],
    }


def realised_schedule_metrics(
    solution: Mapping[str, Any],
    realised_prices: np.ndarray,
    perfect_information: Mapping[str, Any],
    spec: BatterySpec,
) -> dict[str, float]:
    charge = np.asarray(solution["charge"], dtype=float)
    discharge = np.asarray(solution["discharge"], dtype=float)
    energy = np.asarray(solution["energy"], dtype=float)
    prices = np.asarray(realised_prices, dtype=float)
    dt = spec.interval_hours
    market_cost = float(dt * np.sum(prices * (charge - discharge)))
    wear_quantity_mwh = float(dt * np.sum(discharge))
    degradation_cost = spec.degradation_cny_per_mwh * wear_quantity_mwh
    realised_cost = market_cost + degradation_cost
    pi_cost = float(perfect_information["perfect_information_cost"])
    raw_regret = realised_cost - pi_cost
    if raw_regret < -spec.perfect_information_tolerance_cny:
        raise RuntimeError(
            f"realised cost beats perfect information by {-raw_regret:.6g} CNY, "
            f"beyond tolerance {spec.perfect_information_tolerance_cny:.6g}"
        )
    return {
        "realised_market_cost": market_cost,
        "discharged_mwh": wear_quantity_mwh,
        "realised_degradation_cost": float(degradation_cost),
        "realised_cost": realised_cost,
        "realised_profit": -realised_cost,
        "perfect_information_cost": pi_cost,
        "perfect_information_profit": -pi_cost,
        "raw_regret": raw_regret,
        "regret": max(0.0, raw_regret),
        "charge_mwh": float(dt * charge.sum()),
        "discharge_mwh": float(dt * discharge.sum()),
        "throughput_mwh": float(dt * np.sum(charge + discharge)),
        "equivalent_full_cycles": float(
            dt * np.sum(charge + discharge) / (2.0 * spec.energy_mwh)
        ),
        "minimum_soc_mwh": float(min(spec.initial_energy_mwh, energy.min())),
        "maximum_soc_mwh": float(max(spec.initial_energy_mwh, energy.max())),
        "max_simultaneous_charge_discharge_mw": float(
            np.minimum(charge, discharge).max()
        ),
        "maximum_energy_equality_residual": float(
            solution["maximum_energy_equality_residual"]
        ),
        "maximum_shared_inverter_violation": float(
            solution["maximum_shared_inverter_violation"]
        ),
        **cycle_cap_metrics(charge, discharge, spec),
        "perfect_information_max_simultaneous_charge_discharge_mw": float(
            perfect_information[
                "perfect_information_max_simultaneous_charge_discharge_mw"
            ]
        ),
        "perfect_information_maximum_energy_equality_residual": float(
            perfect_information[
                "perfect_information_maximum_energy_equality_residual"
            ]
        ),
        "perfect_information_maximum_shared_inverter_violation": float(
            perfect_information[
                "perfect_information_maximum_shared_inverter_violation"
            ]
        ),
        "perfect_information_cell_side_equivalent_full_cycles": float(
            perfect_information[
                "perfect_information_cell_side_equivalent_full_cycles"
            ]
        ),
        "perfect_information_cell_side_efc_cap_slack": float(
            perfect_information["perfect_information_cell_side_efc_cap_slack"]
        ),
        "perfect_information_cell_side_efc_cap_violation": float(
            perfect_information[
                "perfect_information_cell_side_efc_cap_violation"
            ]
        ),
    }


def aggregate_results(daily: pd.DataFrame) -> pd.DataFrame:
    grouped = daily.groupby(["split", "method", "source_type"], as_index=False).agg(
        n_days=("delivery_day", "nunique"),
        mean_realised_cost=("realised_cost", "mean"),
        total_realised_cost=("realised_cost", "sum"),
        mean_realised_profit=("realised_profit", "mean"),
        total_realised_profit=("realised_profit", "sum"),
        mean_perfect_information_cost=("perfect_information_cost", "mean"),
        total_perfect_information_profit=("perfect_information_profit", "sum"),
        mean_regret=("regret", "mean"),
        total_regret=("regret", "sum"),
        median_regret=("regret", "median"),
        regret_std=("regret", "std"),
        mean_forecast_objective=("forecast_objective", "mean"),
        mean_realised_degradation_cost=("realised_degradation_cost", "mean"),
        mean_charge_mwh=("charge_mwh", "mean"),
        mean_discharge_mwh=("discharge_mwh", "mean"),
        mean_equivalent_full_cycles=("equivalent_full_cycles", "mean"),
        mean_cell_side_equivalent_full_cycles=(
            "cell_side_equivalent_full_cycles",
            "mean",
        ),
        maximum_cell_side_efc_cap_violation=(
            "cell_side_efc_cap_violation",
            "max",
        ),
    )
    grouped["rank_mean_realised_cost"] = (
        grouped.groupby("split")["mean_realised_cost"]
        .rank(method="min")
        .astype(int)
    )
    grouped["rank_mean_regret"] = (
        grouped.groupby("split")["mean_regret"].rank(method="min").astype(int)
    )
    grouped["profit_capture_ratio"] = np.divide(
        grouped["total_realised_profit"],
        grouped["total_perfect_information_profit"],
        out=np.full(len(grouped), np.nan, dtype=float),
        where=np.abs(grouped["total_perfect_information_profit"].to_numpy()) > 1e-12,
    )
    return grouped.sort_values(
        ["split", "rank_mean_realised_cost", "method"], kind="stable"
    ).reset_index(drop=True)
