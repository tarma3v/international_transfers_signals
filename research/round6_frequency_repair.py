"""Frozen frequency repair for the round-five reset-Hist/anchor blend."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _bootstrap_all, _fired
from research.round5_features import load_round5_features
from research.round5_refit_calibration import _evaluate


OUT = Path("results/research/round6/frequency_repair")
SOURCE = Path("results/research/round5/adaptation/outputs.pkl")
CANDIDATE = "quarterly_reset_hist_anchor50"
POLICY = (.20, 250, 0)


def _breakdown(output, years, y, benefit, dates, currencies):
    rate, rolling, cooldown = POLICY
    valid, fired = _fired(
        output, years, dates, currencies, y, rate, rolling, cooldown,
    )
    groups = [("overall", "all", valid, len(CORRIDORS))]
    for year in years:
        groups.append((
            "year", str(year), valid & np.asarray([day.year == year for day in dates]),
            len(CORRIDORS),
        ))
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                day.year == year and (day.month - 1) // 3 + 1 == quarter
                for day in dates
            ])
            if scope.any():
                groups.append(("quarter", f"{year}Q{quarter}", scope, len(CORRIDORS)))
    for currency in CORRIDORS:
        groups.append(("currency", currency, valid & (currencies == currency), 1))
    rows = []
    for kind, group, scope, n_corridors in groups:
        active = scope & fired
        values = benefit[active & ~np.isnan(benefit)]
        base = float(y[scope].mean())
        hit = float(y[active].mean()) if active.any() else np.nan
        rows.append({
            "breakdown": kind, "group": group,
            "n_scope": int(scope.sum()), "n_signals": int(active.sum()),
            "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
            "base_rate": base, "hit_rate": hit,
            "lift": hit / base if active.any() else np.nan,
            "forward_benefit_bps": float(np.mean(values)) if len(values) else np.nan,
        })
    return pd.DataFrame(rows), valid, fired


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    with SOURCE.open("rb") as handle:
        output = pickle.load(handle)[CANDIDATE]

    rows = []
    for period, years in (("screen_2024", (2024,)),
                          ("confirmation_2025", (2025,)),
                          ("audit_2026", (2026,)),
                          ("combined_2025_2026", (2025, 2026))):
        item = _evaluate(output, years, *POLICY, y, benefit, dates, currencies)
        item.update({"period": period, "candidate": CANDIDATE,
                     "rate": POLICY[0], "rolling": POLICY[1],
                     "cooldown": POLICY[2]})
        rows.append(item)
    pd.DataFrame(rows).to_csv(OUT / "selected_policy_results.csv", index=False)

    breakdown, valid, fired = _breakdown(
        output, (2025, 2026), y, benefit, dates, currencies,
    )
    breakdown.to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    draws = _bootstrap_all(y, benefit, dates, valid, {CANDIDATE: fired})[CANDIDATE]
    lift = draws["lift"][np.isfinite(draws["lift"])]
    gain = draws["benefit"][np.isfinite(draws["benefit"])]
    pd.DataFrame([{
        "candidate": CANDIDATE,
        "lift_ci_low": float(np.quantile(lift, .025)),
        "lift_ci_high": float(np.quantile(lift, .975)),
        "p_lift_le_1": float((np.sum(lift <= 1) + 1) / (len(lift) + 1)),
        "benefit_ci_low": float(np.quantile(gain, .025)),
        "benefit_ci_high": float(np.quantile(gain, .975)),
    }]).to_csv(OUT / "block_bootstrap.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "candidate": CANDIDATE, "policy": POLICY,
        "policy_selected_on": 2024, "next_rate_feature": False,
        "confirmation": 2025, "audit": 2026,
        "pristine_holdout_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[[
        "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "quarter_frequency_min", "quarter_frequency_max",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
