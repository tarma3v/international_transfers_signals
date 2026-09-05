"""Packet-W low-dose consensus with the shared five-horizon experts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap, _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/lowdose_horizon")
PRIMARY_SOURCE = Path("results/research/round6/multiobjective_blend/outputs.pkl")
HORIZON_SOURCE = Path("results/research/round6/shared_horizon/outputs.pkl")
EXPERTS = (
    "shared_extra_minimum", "shared_extra_geomean",
    "shared_extra_conservative", "shared_hist_minimum",
    "shared_xgb_conservative",
)
PRIMARY_WEIGHTS = (.80, .75, 2.0 / 3.0)


def main() -> None:
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
    with PRIMARY_SOURCE.open("rb") as handle:
        primary = pickle.load(handle)["stack50_benefit50"]
    with HORIZON_SOURCE.open("rb") as handle:
        horizon = pickle.load(handle)

    outputs = {}
    for expert in EXPERTS:
        for primary_weight in PRIMARY_WEIGHTS:
            suffix = int(round(primary_weight * 100))
            name = f"primary{suffix:02d}_{expert}"
            outputs[name] = combine_causal(
                [primary, horizon[expert]],
                (primary_weight, 1.0 - primary_weight), dates, currencies,
            )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    policies = [row for row in _policy_rows() if row["policy_type"] == "rolling"]
    screen_rows = []
    for candidate, output in outputs.items():
        for policy in policies:
            item = _evaluate(output, (2024,), policy, y, benefit, dates, currencies)
            item.update({"candidate": candidate, **policy})
            screen_rows.append(item)
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(OUT / "screen_2024_grid.csv", index=False)
    selected = pd.DataFrame([_choose(part) for _, part in screen.groupby("candidate")])
    selected = selected.sort_values(["robustness", "lift"], ascending=False)
    selected.to_csv(OUT / "screen_2024_selected.csv", index=False)

    later_rows = []
    for row in selected.itertuples(index=False):
        policy = _row_policy(row)
        for period, years in (
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(outputs[row.candidate], years, policy,
                             y, benefit, dates, currencies)
            item.update({"period": period, "candidate": row.candidate, **policy})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            later_rows.append(item)
    later = pd.DataFrame(later_rows)
    later.to_csv(OUT / "later_results.csv", index=False)

    finalists = selected.head(8)
    boot_2025, masks_2025, valid_2025 = _bootstrap(
        finalists, outputs, (2025,), y, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        finalists, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    boot_both["period"] = "2025_2026"
    pd.concat([boot_2025, boot_both], ignore_index=True).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    pd.concat([
        _circular_shift_audit(
            y, dates, currencies, valid_2025, masks_2025, "retrospective_2025",
        ),
        _circular_shift_audit(
            y, dates, currencies, valid_both, masks_both,
            "retrospective_2025_2026",
        ),
    ], ignore_index=True).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown_rows = []
    for row in finalists.itertuples(index=False):
        breakdown_rows.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "finalist_breakdown.csv", index=False)

    (OUT / "protocol.json").write_text(json.dumps({
        "primary": "stack50_benefit50", "experts": EXPERTS,
        "primary_weights": PRIMARY_WEIGHTS,
        "combination": "fixed causal per-currency score ranks",
        "weights_fitted": False, "policy_selected_on": 2024,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n2024\n" + selected[[
        "candidate", "rate", "rolling", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].to_string(index=False))
    print("\nLATER\n" + later[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "corridor_lift_min", "quarter_frequency_min",
        "quarter_frequency_max", "robustness",
    ]].sort_values(
        ["period", "robustness", "lift"], ascending=[True, False, False],
    ).to_string(index=False))


if __name__ == "__main__":
    main()
