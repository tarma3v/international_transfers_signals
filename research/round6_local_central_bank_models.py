"""Packet-CB: strictly lagged local-central-bank shadow-rate experiment."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_local_central_bank_features import (
    build_nbt_features,
    causality_check,
    load_nbt,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/local_central_bank_models")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
ROUTER = Path("results/research/round6/cny_expert_router/outputs.pkl")
ORDER = (
    "nbt_direct_basis",
    "nbt_usd_cross_basis",
    "nbt_cny_cross_basis",
    "nbt_consensus_basis",
    "nbt_direct_consensus_mean",
    "nbt_direct_stale20",
    "primary_tjs_nbt_direct",
    "router_tjs_nbt_direct",
    "primary75_nbt25",
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _overlay(base_score, local_score, currencies, target="TJS"):
    result = np.asarray(base_score, dtype=float).copy()
    rows = currencies == target
    result[rows] = local_score[rows]
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    nbt, digest = load_nbt()
    matrix, names = build_nbt_features(index, series, references, nbt)
    if not causality_check(index, series, references, nbt):
        raise AssertionError("NBT physical causality check failed")
    col = {name: matrix[:, names.index(name)].astype(float) for name in names}
    direct = col["nbt_direct_basis_bps"]
    consensus = col["nbt_consensus_basis_bps"]
    raw = {
        "nbt_direct_basis": direct,
        "nbt_usd_cross_basis": col["nbt_usd_basis_bps"],
        "nbt_cny_cross_basis": col["nbt_cny_basis_bps"],
        "nbt_consensus_basis": consensus,
        "nbt_direct_consensus_mean": .50 * direct + .50 * consensus,
        "nbt_direct_stale20": delayed_by_currency(
            direct[:, None], index, rows=20,
        )[:, 0],
    }
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    outputs = {name: _outputs(score, y, dates) for name, score in raw.items()}
    primary = _load(PRIMARY)["logit50_extra50"]
    router = _load(ROUTER)["router_tree_hard"]
    primary_score = row_scores(primary, len(y))
    router_score = row_scores(router, len(y))
    outputs["primary_tjs_nbt_direct"] = _outputs(
        _overlay(primary_score, direct, currencies), y, dates,
    )
    outputs["router_tjs_nbt_direct"] = _outputs(
        _overlay(router_score, direct, currencies), y, dates,
    )
    outputs["primary75_nbt25"] = combine_causal(
        [primary, outputs["nbt_direct_basis"]], (.75, .25), dates, currencies,
    )
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
        y, dates, currencies, valid, masks, "local_central_bank_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CB",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "source": "National Bank of Tajikistan official dynamic XML archive",
        "source_files": [
            str(path) for path in sorted(Path("data").glob("external_nbt_*_2016_2026.xml"))
        ],
        "payload_sha256": digest,
        "asof_rule": "NBT effective date strictly before signal date; CBR date <= signal date",
        "publication_time_assumed": False,
        "physical_future_corruption_check": True,
        "stale_control_rows_per_currency": 20,
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
