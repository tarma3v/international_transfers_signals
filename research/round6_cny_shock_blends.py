"""Packet-AV fixed low-dose non-market complements for the 2022 shock bridge."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_lifecycle import YEARS, _stitch
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_shock_blends")
PRE = Path("results/research/round6/cny_pre2022/outputs.pkl")
BRIDGE = Path("results/research/round6/cny_shock_bridge/outputs.pkl")
LATER = Path("results/research/round6/cny_history_weighting/outputs.pkl")
ANCHOR = Path("results/research/candidate_outputs_h5_v2.pkl")
GLOBAL = Path("results/research/round2/diverse_outputs.pkl")
ROUTER = Path("results/research/round2/router_outputs.pkl")
SHOCK_YEARS = (2022, 2023)
ORDER = (
    "cny_expanding",
    "cny75_anchor25",
    "cny75_global_extra25",
    "cny75_regime_soft25",
    "cny_anchor_global_equal",
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    pre = _load(PRE)["logit50_extra50"]
    cny = _load(BRIDGE)["expanding_consensus"]
    later = _load(LATER)["hard_reset"]
    anchor = _load(ANCHOR)["anchor_multiscale"]
    global_extra = _load(GLOBAL)["global_compact_extra"]
    regime = _load(ROUTER)["regime_soft"]
    shock_outputs = {
        "cny_expanding": cny,
        "cny75_anchor25": combine_causal(
            [cny, anchor], (.75, .25), dates, currencies,
        ),
        "cny75_global_extra25": combine_causal(
            [cny, global_extra], (.75, .25), dates, currencies,
        ),
        "cny75_regime_soft25": combine_causal(
            [cny, regime], (.75, .25), dates, currencies,
        ),
        "cny_anchor_global_equal": combine_causal(
            [cny, anchor, global_extra], (1 / 3, 1 / 3, 1 / 3),
            dates, currencies,
        ),
    }
    lifecycle_outputs = {
        name: _stitch(pre, shock_outputs[name], later) for name in ORDER
    }
    with (OUT / "shock_outputs.pkl").open("wb") as handle:
        pickle.dump(shock_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUT / "lifecycle_outputs.pkl").open("wb") as handle:
        pickle.dump(lifecycle_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    shock_rows, shock_breakdown = [], []
    lifecycle_rows, lifecycle_breakdown = [], []
    for candidate in ORDER:
        for period, years in (
            ("2022", (2022,)), ("2023", (2023,)),
            ("2022_2023", SHOCK_YEARS),
        ):
            item = _evaluate(
                shock_outputs[candidate], years, POLICY,
                y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            shock_rows.append(item)
        shock_breakdown.extend(_breakdown(
            candidate, shock_outputs[candidate], SHOCK_YEARS, POLICY,
            y, benefit, dates, currencies,
        ))
        item = _evaluate(
            lifecycle_outputs[candidate], YEARS, POLICY,
            y, benefit, dates, currencies,
        )
        item.update({"candidate": candidate, "period": "2017_2026", **POLICY})
        lifecycle_rows.append(item)
        lifecycle_breakdown.extend(_breakdown(
            candidate, lifecycle_outputs[candidate], YEARS, POLICY,
            y, benefit, dates, currencies,
        ))

    shock_results = pd.DataFrame(shock_rows)
    shock_detail = pd.DataFrame(shock_breakdown)
    quarter_min = (
        shock_detail[shock_detail.breakdown == "quarter"]
        .groupby("candidate")
        .agg(shock_quarter_lift_min=("lift", "min"),
             shock_quarter_rate_min=("frequency", "min"),
             shock_quarter_benefit_min=("forward_benefit_bps", "min"))
        .reset_index()
    )
    shock_results = shock_results.merge(quarter_min, on="candidate", how="left")
    shock_results["predeclared_order"] = shock_results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    lifecycle_results = pd.DataFrame(lifecycle_rows)
    lifecycle_results["predeclared_order"] = lifecycle_results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    shock_results.to_csv(OUT / "shock_results.csv", index=False)
    shock_detail.to_csv(OUT / "shock_breakdown.csv", index=False)
    lifecycle_results.to_csv(OUT / "lifecycle_results.csv", index=False)
    pd.DataFrame(lifecycle_breakdown).to_csv(
        OUT / "lifecycle_breakdown.csv", index=False,
    )

    bootstrap, masks, valid = _bootstrap(
        lifecycle_results, lifecycle_outputs, YEARS,
        y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2017_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "shock_blend_lifecycle_2017_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    paths = (PRE, BRIDGE, LATER, ANCHOR, GLOBAL, ROUTER)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AV",
        "variants": ORDER,
        "shock_years": SHOCK_YEARS,
        "lifecycle_years": YEARS,
        "fixed_policy": POLICY,
        "low_dose_weights": [0.75, 0.25],
        "equal_weights": [1 / 3, 1 / 3, 1 / 3],
        "model_refit": False,
        "source_output_sha256": {str(path): _digest(path) for path in paths},
        "later_period_status": "retrospective fixed-blend shock audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    shock_columns = [
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min",
        "shock_quarter_lift_min", "shock_quarter_rate_min",
    ]
    lifecycle_columns = [
        "predeclared_order", "candidate", "frequency", "lift",
        "forward_benefit_bps", "year_lift_min", "year_frequency_min",
        "corridor_lift_min", "quarter_frequency_min",
    ]
    print("SHOCK\n" + shock_results[shock_columns].sort_values(
        ["period", "predeclared_order"],
    ).to_string(index=False))
    print("\nLIFECYCLE\n" + lifecycle_results[lifecycle_columns].sort_values(
        "predeclared_order",
    ).to_string(index=False))


if __name__ == "__main__":
    main()
