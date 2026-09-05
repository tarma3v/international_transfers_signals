"""Packet-Y business-preserving consensus with shared-horizon experts."""
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
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/business_consensus")
BASE_SOURCE = Path("results/research/round6/multiobjective_blend/outputs.pkl")
HORIZON_SOURCE = Path("results/research/round6/shared_horizon/outputs.pkl")
BASES = {
    "benefit_ranker_anchor25": (.22, 60),
    "stack25_benefit75": (.20, 60),
}
EXPERTS = (
    "shared_extra_minimum", "shared_extra_geomean",
    "shared_extra_conservative", "shared_xgb_conservative",
)
BASE_WEIGHTS = (.80, .75)


def policy(rate, rolling):
    return {
        "policy_type": "rolling", "rate": rate, "rolling": rolling,
        "cooldown": 0, "history": 0, "strong": 0.0, "late": 0.0,
        "late_weekday": 0, "weekly_cap": 0,
    }


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
    with BASE_SOURCE.open("rb") as handle:
        bases = pickle.load(handle)
    with HORIZON_SOURCE.open("rb") as handle:
        horizons = pickle.load(handle)

    outputs, policies = {}, {}
    for base, (rate, rolling) in BASES.items():
        for expert in EXPERTS:
            for base_weight in BASE_WEIGHTS:
                name = f"{base}{int(base_weight*100):02d}_{expert}"
                outputs[name] = combine_causal(
                    [bases[base], horizons[expert]],
                    (base_weight, 1.0 - base_weight), dates, currencies,
                )
                policies[name] = policy(rate, rolling)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for candidate, output in outputs.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(output, years, policies[candidate],
                             y, benefit, dates, currencies)
            item.update({"period": period, "candidate": candidate,
                         **policies[candidate]})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            rows.append(item)
    results = pd.DataFrame(rows)
    order = (
        results[results.period == "screen_2024"]
        .sort_values(["robustness", "lift"], ascending=False).candidate.tolist()
    )
    results["screen_rank"] = results.candidate.map({name: i + 1 for i, name in enumerate(order)})
    results.to_csv(OUT / "results.csv", index=False)

    finalist_names = order[:8]
    selected = results[
        (results.period == "screen_2024") & results.candidate.isin(finalist_names)
    ].sort_values("screen_rank")
    boot_2025, masks_2025, valid_2025 = _bootstrap(
        selected, outputs, (2025,), y, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        selected, outputs, (2025, 2026), y, benefit, dates, currencies,
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
    for candidate in finalist_names:
        breakdown_rows.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), policies[candidate],
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown_rows).to_csv(OUT / "finalist_breakdown.csv", index=False)

    (OUT / "protocol.json").write_text(json.dumps({
        "bases": BASES, "experts": EXPERTS, "base_weights": BASE_WEIGHTS,
        "base_policies_preserved": True, "weights_fitted": False,
        "candidate_order_selected_on": 2024, "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "screen_rank", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_freq_min", "corridor_lift_min",
        "quarter_frequency_min", "quarter_frequency_max", "robustness",
    ]].sort_values(["period", "screen_rank"]).to_string(index=False))


if __name__ == "__main__":
    main()
