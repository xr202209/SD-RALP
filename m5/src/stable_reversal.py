"""Stable-reversal subset of the M5 panel, selected from training outcomes only.

The 27 training origins with complete outcomes before validation are split into
two blocks of 12 origins, separated by three origins.  A series is retained if
the same component has the lowest mean centre-region twCRPS in both blocks, the
same component has the lowest mean upper-region twCRPS in both blocks, and the
two components differ.
"""
import numpy as np
import pandas as pd

def select_stable_reversals(records, model_names, train_origins, first_validation_origin, horizon):
    eligible = [r for r in records if r.day in train_origins and r.day + horizon - 1 < first_validation_origin]
    days = sorted({r.day for r in eligible})
    assert len(days) == 27
    early, late = set(days[:12]), set(days[-12:])
    assert max(early) + horizon - 1 < min(late)
    rows = []
    by_series = {}
    for r in eligible:
        by_series.setdefault(r.series_id, []).append(r)
    for sid, recs in sorted(by_series.items()):
        assert len(recs) == 27 and len({r.day for r in recs}) == 27
        row = {"series_id": sid, "n_selection_origins": 24,
               "selection_last_outcome_day": max(days) + horizon - 1,
               "early_last_outcome_day": max(early) + horizon - 1,
               "late_first_origin": min(late),
               "train_upper_event_count": sum(r.obs_region == "U" for r in recs),
               "train_upper_occupancy": np.mean([r.obs_region == "U" for r in recs])}
        winners = []
        for label, block in (("early", early), ("late", late)):
            subset = [r for r in recs if r.day in block]
            c = np.mean([r.model_c for r in subset], axis=0)
            u = np.mean([r.model_u for r in subset], axis=0)
            assert np.isfinite(c).all() and np.isfinite(u).all()
            ci, ui = int(c.argmin()), int(u.argmin())
            unique = np.count_nonzero(np.isclose(c, c[ci], rtol=1e-10, atol=1e-12)) == 1
            unique &= np.count_nonzero(np.isclose(u, u[ui], rtol=1e-10, atol=1e-12)) == 1
            winners.append((ci, ui, unique))
            row.update({f"{label}_centre_best": model_names[ci], f"{label}_upper_best": model_names[ui],
                        f"{label}_upper_gain": float(u[ci]-u[ui]),
                        f"{label}_centre_gain": float(c[ui]-c[ci]),
                        f"{label}_upper_events": sum(r.obs_region == "U" for r in subset)})
        a, b = winners
        stable = a[:2] == b[:2] and a[0] != a[1] and a[2] and b[2]
        row["stable_reversal"] = bool(stable)
        row["group"] = f"{model_names[a[0]]}__{model_names[a[1]]}" if stable else "not_stable"
        rows.append(row)
    return pd.DataFrame(rows)
