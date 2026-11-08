"""Quantile-average pool: equal-weight average of the component inverse CDFs.

CDFs are piecewise linear on their supplied grid. Repeated probability knots
retain jumps in the inverse CDF; repeated demand knots retain atoms. The PDF
used for the log score remains a grid approximation.
"""
from dataclasses import dataclass

import numpy as np

import regional_pool as base


def inverse(grid, cdf, probabilities, side="left"):
    p = np.asarray(probabilities, dtype=float)
    idx = np.searchsorted(cdf, p, side=side)
    hi = np.clip(idx, 1, len(grid) - 1)
    lo = hi - 1
    width = cdf[hi] - cdf[lo]
    fraction = np.divide(p - cdf[lo], width, out=np.ones_like(p), where=width > 0)
    value = grid[lo] + fraction * (grid[hi] - grid[lo])
    return np.where(idx == 0, grid[0], np.where(idx >= len(grid), grid[-1], value))


@dataclass
class QuantileAverage:
    grid: np.ndarray
    component_cdfs: np.ndarray
    probabilities: np.ndarray
    demands: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray

    def quantile(self, probabilities):
        return np.mean([inverse(self.grid, c, probabilities) for c in self.component_cdfs], axis=0)

    def cvar(self, tau):
        keep = self.probabilities > tau
        p = np.r_[tau, self.probabilities[keep]]
        # The right limit starts the integral when tau lies at an inverse jump.
        right = np.mean([inverse(self.grid, c, [tau], side="right")[0] for c in self.component_cdfs])
        q = np.r_[right, self.demands[keep]]
        return float(base.trapz(q, p) / (1 - tau))


def distribution(rec):
    grid = np.asarray(rec.grid, dtype=float)
    cdfs = np.asarray(rec.comp_cdfs, dtype=float)
    if not np.isfinite(cdfs).all() or np.any(np.diff(cdfs, axis=1) < -1e-12):
        raise ValueError("Invalid component CDF")
    if not np.allclose(cdfs[:, -1], 1.0):
        raise ValueError("Component CDF must end at one")
    knots = np.unique(np.r_[0.0, cdfs.ravel(), 1.0])
    left = np.mean([inverse(grid, c, knots) for c in cdfs], axis=0)
    right = np.mean([inverse(grid, c, knots, side="right") for c in cdfs], axis=0)
    right[-1] = left[-1]
    p = np.repeat(knots, 2)
    q = np.column_stack([left, right]).ravel()
    q = np.maximum.accumulate(q)
    cdf = np.interp(grid, q, p, left=0.0, right=1.0)
    # Grid density, used only for the log score.
    pdf = np.maximum(np.gradient(cdf, grid), 0.0)
    pdf[0] += 2.0 * cdf[0] / (grid[1] - grid[0])
    pdf /= max(base.trapz(pdf, grid), 1e-12)
    return QuantileAverage(grid, cdfs, p, q, pdf, cdf)


def prediction_row(rec):
    dist = distribution(rec)
    row = base.prediction_row(rec, dist.pdf, dist.cdf)
    qs = dist.quantile([.10, .50, .90, .95, .99])
    row.update(dict(zip(["q10", "q50", "q90", "q95", "q99"], np.maximum(qs, 0.0))))
    q_tau = float(max(0.0, dist.quantile([rec.tau])[0]))
    row.update(q_tau=q_tau, newsvendor_cost=base.newsvendor_cost(q_tau, rec.y, rec.tau),
               stockout=float(rec.y > q_tau), cvar_tau=dist.cvar(rec.tau))
    return row
