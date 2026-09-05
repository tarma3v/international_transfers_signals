"""Packet-CC: orientation and dynamics screen for strictly lagged NBT data.

This is explicitly exploratory after packet CB showed that the intuitive
positive level orientation did not transport.  Formula choice is nevertheless
made on 2024 only before reporting 2025-2026.
"""
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
from research.round6_cny_decomposition import POLICY
from research.round6_local_central_bank_features import build_nbt_features, load_nbt
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/local_central_bank_orientation")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
SCREEN_ORDER = (
    "negative_direct_basis",
    "negative_usd_basis",
    "negative_cny_basis",
    "negative_consensus_basis",
    "inverse_rub_momentum_1",
    "inverse_rub_momentum_2",
    "inverse_rub_momentum_5",
    "inverse_rub_momentum_mean",
    "direct_usd_disagreement",
    "direct_cny_disagreement",
    "usd_cny_disagreement",
)


def _screen_choice(results):
    part = results[results.period == "screen_2024"].copy()
    feasible = part[
        part.frequency.between(1.0, 2.0)
        & part.quarter_frequency_min.ge(.70)
        & part.forward_benefit_bps.gt(0.0)
    ].copy()
    pool = feasible if len(feasible) else part.copy()
    pool["robustness"] = pool[["lift", "corridor_lift_min"]].min(axis=1)
    return str(pool.sort_values(
        ["robustness", "lift", "forward_benefit_bps"], ascending=False,
    ).iloc[0].candidate)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    nbt, digest = load_nbt()
    matrix, names = build_nbt_features(index, series, references, nbt)
    col = {name: matrix[:, names.index(name)].astype(float) for name in names}
    raw = {
        "negative_direct_basis": -col["nbt_direct_basis_bps"],
        "negative_usd_basis": -col["nbt_usd_basis_bps"],
        "negative_cny_basis": -col["nbt_cny_basis_bps"],
        "negative_consensus_basis": -col["nbt_consensus_basis_bps"],
        "inverse_rub_momentum_1": -col["nbt_rub_quote_ret_1"],
        "inverse_rub_momentum_2": -col["nbt_rub_quote_ret_2"],
        "inverse_rub_momentum_5": -col["nbt_rub_quote_ret_5"],
        "inverse_rub_momentum_mean": -(
            col["nbt_rub_quote_ret_1"]
            + col["nbt_rub_quote_ret_2"]
            + col["nbt_rub_quote_ret_5"]
        ) / 3.0,
        "direct_usd_disagreement": col["nbt_direct_minus_usd_bps"],
        "direct_cny_disagreement": col["nbt_direct_minus_cny_bps"],
        "usd_cny_disagreement": col["nbt_usd_minus_cny_bps"],
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
    screen_rows = []
    for candidate in SCREEN_ORDER:
        item = _evaluate(
            outputs[candidate], (2024,), POLICY, y, benefit, dates, currencies,
        )
        item.update({"candidate": candidate, "period": "screen_2024", **POLICY})
        screen_rows.append(item)
    screen = pd.DataFrame(screen_rows)
    chosen = _screen_choice(screen)
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    for weight in (.10, .25, .40):
        name = f"primary{int((1-weight)*100)}_{chosen}{int(weight*100)}"
        outputs[name] = combine_causal(
            [primary, outputs[chosen]], (1.0 - weight, weight), dates, currencies,
        )
    order = SCREEN_ORDER + tuple(
        f"primary{int((1-weight)*100)}_{chosen}{int(weight*100)}"
        for weight in (.10, .25, .40)
    )
    rows = []
    for candidate in order:
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
        {name: i + 1 for i, name in enumerate(order)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    later_screen = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(
        later_screen, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "local_cb_orientation_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in order:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CC", "variants": order, "fixed_policy": POLICY,
        "status": "exploratory orientation screen after packet CB",
        "selection_period": 2024, "selected_formula": chosen,
        "blend_weights": [.10, .25, .40],
        "payload_sha256": digest,
        "asof_rule": "NBT effective date strictly before signal date; CBR date <= signal date",
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected on 2024: {chosen}\n")
    display = results.sort_values(["period", "predeclared_order"])
    print(display[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min", "robustness",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
