"""Weekly-budget audit of the frozen packet-D external challenger."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round5_features import load_round5_features
from research.round6_rate_control import Policy, _breakdown, _metrics


OUT = Path("results/research/round6/external_weekly")
SOURCE = Path("results/research/round6/external_reset")
CANDIDATE = "macro_full_hist_anchor50"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(SOURCE / "screen_2024_grid.csv")
    pool = grid[(grid.candidate == CANDIDATE) & (grid.policy_type == "weekly")].copy()
    feasible = pool[
        pool.frequency.between(1.00, 2.00)
        & pool.corridor_freq_min.ge(.80)
        & pool.quarter_frequency_min.ge(.70)
        & pool.forward_benefit_bps.gt(0)
    ].copy()
    feasible["robustness"] = feasible[["lift", "corridor_lift_min"]].min(axis=1)
    chosen = feasible.sort_values(["robustness", "lift"], ascending=False).iloc[0]
    chosen.to_frame().T.to_csv(OUT / "selected_2024.csv", index=False)
    policy = Policy(
        int(chosen.history), float(chosen.strong), float(chosen.late),
        int(chosen.late_weekday), int(chosen.weekly_cap),
    )

    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    with (SOURCE / "outputs.pkl").open("rb") as handle:
        output = pickle.load(handle)[CANDIDATE]
    rows = []
    for period, years in (("screen_2024", (2024,)),
                          ("confirmation_2025", (2025,)),
                          ("audit_2026", (2026,)),
                          ("combined_2025_2026", (2025, 2026))):
        item = _metrics(output, years, policy, y, benefit, dates, currencies)
        item.update({"period": period, "candidate": CANDIDATE,
                     **policy.__dict__})
        rows.append(item)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(_breakdown(
        CANDIDATE, output, (2025, 2026), policy, y, benefit, dates, currencies,
    )).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "candidate": CANDIDATE, "policy": policy.__dict__,
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
