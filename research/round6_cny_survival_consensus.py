"""Packet-BV fixed consensus using the causal survival expert."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_survival_consensus")
SURVIVAL = Path("results/research/round6/cny_survival_hazard/outputs.pkl")
SHADOW = Path("results/research/round6/cny_shadow_nowcast/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
ORDER = (
    "primary75_survival25",
    "shadow75_survival25",
    "primary50_survival25_shadow25",
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


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
    survival = _load(SURVIVAL)["survival_cumulative_geometric"]
    shadow = _load(SHADOW)["shadow_close_basis"]
    primary = _load(PRIMARY)["logit50_extra50"]
    outputs = {
        "primary75_survival25": combine_causal(
            [primary, survival], (.75, .25), dates, currencies,
        ),
        "shadow75_survival25": combine_causal(
            [shadow, survival], (.75, .25), dates, currencies,
        ),
        "primary50_survival25_shadow25": combine_causal(
            [primary, survival, shadow], (.50, .25, .25), dates, currencies,
        ),
    }
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for candidate in ORDER:
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                outputs[candidate], years, POLICY, y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    screen = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(
        screen, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "cny_survival_consensus_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BV",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "screen_year": 2024,
        "screened_weights": [
            [.25, .75], [.50, .50], [.75, .25],
            [1 / 3, 1 / 3, 1 / 3], [.50, .25, .25],
        ],
        "combination": "fixed per-currency causal score ranks",
        "components_refitted": False,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))


if __name__ == "__main__":
    main()
