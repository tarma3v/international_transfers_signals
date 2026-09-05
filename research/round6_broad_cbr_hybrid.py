"""Packet-I score blends: broad high-confidence plus stable CBR baseload."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.model_study import combine_outputs
from research.round5_features import load_round5_features
from research.round6_resolved_models import (
    _breakdown, _choose, _evaluate, _policy_rows, _row_policy,
)


OUT = Path("results/research/round6/broad_cbr_hybrid")
SOURCE = Path("results/research/round6/broad_cbr/outputs.pkl")
BROAD = "broad_full_extra"
BASELOAD = "packet_e_cbr_anchor50"
WEIGHTS = (.25, .50, .75)


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
    with SOURCE.open("rb") as handle:
        all_outputs = pickle.load(handle)
    outputs = {BROAD: all_outputs[BROAD], BASELOAD: all_outputs[BASELOAD]}
    for broad_weight in WEIGHTS:
        name = f"broad{int(broad_weight * 100):02d}_baseload{int((1-broad_weight)*100):02d}"
        outputs[name] = combine_outputs(
            [outputs[BROAD], outputs[BASELOAD]],
            (broad_weight, 1.0 - broad_weight), currencies,
        )

    policies = _policy_rows()
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

    rows = []
    for selected_row in selected.itertuples(index=False):
        policy = _row_policy(selected_row)
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                outputs[selected_row.candidate], years, policy,
                y, benefit, dates, currencies,
            )
            item.update({
                "period": period, "candidate": selected_row.candidate, **policy,
            })
            rows.append(item)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)

    breakdown = []
    for selected_row in selected.itertuples(index=False):
        breakdown.extend(_breakdown(
            selected_row.candidate, outputs[selected_row.candidate],
            (2025, 2026), _row_policy(selected_row), y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT / "protocol.json").write_text(json.dumps({
        "broad_component": BROAD,
        "baseload_component": BASELOAD,
        "broad_weights": WEIGHTS,
        "combination": "per-currency rank blend against prior calibration year",
        "policy_selected_on": 2024,
        "later_period_status": "protocol-controlled retrospective, not pristine",
        "next_rate_feature": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "candidate", "period", "policy_type", "frequency", "lift",
        "forward_benefit_bps", "corridor_freq_min", "corridor_lift_min",
        "quarter_frequency_min", "quarter_frequency_max",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
