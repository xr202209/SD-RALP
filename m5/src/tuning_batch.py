"""Validation scores of the core methods and particle filters, one origin at a time."""
from itertools import groupby

import numpy as np

import regional_pool as base
import benchmarks as bench
import update_timing as timing


class Day:
    def __init__(self, records):
        self.day = records[0].day
        self.n = len(records)
        self.grid = np.stack([r.grid for r in records])
        self.pdf = np.stack([r.comp_pdfs for r in records])
        self.y = np.array([r.y for r in records])
        self.tau = np.array([r.tau for r in records])
        self.low = np.array([r.c_low for r in records])
        self.high = np.array([r.c_high for r in records])
        self.dz = self.grid[:, 1] - self.grid[:, 0]
        self.ml = self.grid <= self.low[:, None]
        self.mu = self.grid > self.high[:, None]
        self.mc = ~self.ml & ~self.mu
        self.indicator = self.y[:, None] <= self.grid
        self.high_idx = np.clip(np.sum(self.grid <= self.high[:, None], axis=1)-1, 0, self.grid.shape[1]-2)
        rr = np.arange(self.n)
        self.high_fraction = (self.high-self.grid[rr, self.high_idx])/self.dz
        self.excess = np.maximum(self.y - self.grid[:, -1], 0)
        self.losses = {r: np.mean([base.regional_model_losses(x, r) for x in records], axis=0)
                       for r in ['global', 'L', 'C', 'U']}
        self.disagreement = {r: np.mean([base.regional_disagreement(x, r) for x in records], axis=0)
                             for r in ['global', 'L', 'C', 'U']}
        self.density = np.stack([bench.component_density_at_y(r) for r in records])
        self.diversity = bench.panel_diversity(records)

    def score(self, weights, region):
        gradient = self.losses[region] - self.disagreement[region] @ weights
        return weights * (weights @ gradient - gradient)

    def metrics(self, wl, wc, wu):
        p = (np.einsum('nkg,k->ng', self.pdf, wl) * self.ml
             + np.einsum('nkg,k->ng', self.pdf, wc) * self.mc
             + np.einsum('nkg,k->ng', self.pdf, wu) * self.mu)
        # The density normalization cancels in the normalized cumulative sum.
        cdf = np.cumsum(np.maximum(p, 0), axis=1)
        cdf /= np.maximum(cdf[:, -1:], 1e-12)
        cdf[:, -1] = 1
        error = (cdf - self.indicator) ** 2
        crps = error.sum(axis=1) * self.dz + self.excess
        tw = (error * self.mu).sum(axis=1) * self.dz + self.excess
        idx = np.sum(cdf < self.tau[:, None], axis=1)
        j = np.clip(idx, 1, cdf.shape[1] - 1)
        rows = np.arange(self.n)
        y0, y1 = cdf[rows, j - 1], cdf[rows, j]
        x0, x1 = self.grid[rows, j - 1], self.grid[rows, j]
        q = np.where(y1 > y0, x0 + (self.tau-y0)*(x1-x0)/np.maximum(y1-y0, 1e-300), x1)
        q[idx == 0] = self.grid[idx == 0, 0]
        high_cdf = cdf[rows, self.high_idx] + self.high_fraction*(cdf[rows, self.high_idx+1]-cdf[rows, self.high_idx])
        prob = 1 - np.maximum(high_cdf, 1e-12)
        brier = ((self.y > self.high) - prob) ** 2
        cost = np.maximum(q-self.y, 0) + self.tau/(1-self.tau)*np.maximum(self.y-q, 0)
        return np.array([crps.sum(), tw.sum(), brier.sum(), cost.sum(), (self.y > q).sum(), (1-self.tau).sum()])


class Panel:
    def __init__(self, warm, score):
        def days(records):
            return [Day(list(g)) for _, g in groupby(sorted(records, key=lambda r: (r.day, r.series_id)), lambda r: r.day)]
        self.warm = days(warm)
        self.score_days = days(score)
        self.n = sum(d.n for d in self.score_days)

    def summary(self, total):
        crps, tw, brier, cost, stockout, target = total / self.n
        return dict(crps=crps, twcrps_u=tw, brier_u=brier, newsvendor_cost=cost,
                    service_level=1-stockout, ecal=abs(stockout-target))

    def core(self, method, static, alpha, beta, rho, horizon):
        tg, tc, tl, tu, og, oc, ol, ou = base.initialize_score_driven_states(static, beta)
        pending = []
        total = np.zeros(6)
        for collect, days in [(False, self.warm), (True, self.score_days)]:
            for d in days:
                available, pending = timing.split_available_updates(pending, d.day, horizon)
                for _, (previous, wl, wc, wu) in available:
                    if method == 'dynamic_global':
                        tg = base.update_dynamic_global_state(tg, og, alpha, beta, previous.score(wl, 'global'))
                    elif method == 'sd_ralp':
                        tc, tl, tu = base.update_sd_ralp_states(tc, tl, tu, oc, ol, ou, alpha, beta, rho,
                            previous.score(wc, 'C'), previous.score(wl, 'L'), previous.score(wu, 'U'))
                if method == 'equal':
                    wl = wc = wu = np.ones(len(tg))/len(tg)
                elif method == 'global':
                    wl = wc = wu = static['global']
                elif method == 'static_ralp':
                    wl, wc, wu = static['L'], static['C'], static['U']
                elif method == 'dynamic_global':
                    wl = wc = wu = base.softmax(tg)
                elif method == 'sd_ralp':
                    wl, wc, wu = base.softmax(tl), base.softmax(tc), base.softmax(tu)
                else:
                    raise ValueError(method)
                if collect:
                    total += d.metrics(wl, wc, wu)
                if method in ['dynamic_global', 'sd_ralp']:
                    pending.append((d.day, (d, wl.copy(), wc.copy(), wu.copy())))
        return self.summary(total)

    def particle(self, spec, diversity, horizon):
        state = bench.initial_particle_state(spec, self.score_days[0].pdf.shape[1])
        pending = []
        total = np.zeros(6)
        for collect, days in [(False, self.warm), (True, self.score_days)]:
            for d in days:
                available, pending = timing.split_available_updates(pending, d.day, horizon)
                idx = 0
                while idx < len(available):
                    bench.apply_delayed_particle_likelihood(state, spec, available[idx][1])
                    pending, remaining = bench.maybe_resample_particles(state, spec, pending, available[idx+1:])
                    available = available[:idx+1] + remaining
                    idx += 1
                state.alpha += state.rng.normal(0, spec.coefficient_noise, size=state.alpha.shape)
                theta = 2*(1/(1+np.exp(-state.alpha))-0.5)
                div = d.diversity if diversity else np.zeros(state.x.shape[1])
                state.x = np.clip(theta[:, 0, None] + theta[:, 1, None]*state.x + theta[:, 2, None]*div[None, :]
                                  + state.rng.normal(0, spec.state_noise, size=state.x.shape), -12, 12)
                weights = bench.row_softmax(state.x)
                effective = bench.stable_softmax(state.log_weights) @ weights
                if collect:
                    total += d.metrics(effective, effective, effective)
                pending.append((d.day, dict(density=d.density, component_weights=weights.copy())))
        return self.summary(total)
