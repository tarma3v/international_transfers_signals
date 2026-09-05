"""Packet-CF: causal Armenian-central-bank shadow-rate models."""
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
from research.round6_armenian_central_bank_features import (
    build_cba_features, causality_check, load_cba,
)
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/armenian_central_bank_models")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
GEOMETRY = Path("results/research/round6/cny_expert_geometry/outputs.pkl")
SCREEN_ORDER = (
    "cba_direct_basis", "cba_negative_direct_basis",
    "cba_usd_basis", "cba_negative_usd_basis",
    "cba_cny_basis", "cba_negative_cny_basis",
    "cba_consensus_basis", "cba_negative_consensus_basis",
    "cba_inverse_rub_momentum_1", "cba_inverse_rub_momentum_2",
    "cba_inverse_rub_momentum_5", "cba_direct_usd_disagreement",
    "cba_direct_cny_disagreement", "cba_usd_cny_disagreement",
    "cba_consensus_stale20",
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _choose(results):
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
    cba, digest = load_cba()
    matrix, names = build_cba_features(index, series, references, cba)
    causality_check(index, series, references, cba)
    col = {name: matrix[:, names.index(name)].astype(float) for name in names}
    raw = {
        "cba_direct_basis": col["cba_direct_basis_bps"],
        "cba_negative_direct_basis": -col["cba_direct_basis_bps"],
        "cba_usd_basis": col["cba_usd_basis_bps"],
        "cba_negative_usd_basis": -col["cba_usd_basis_bps"],
        "cba_cny_basis": col["cba_cny_basis_bps"],
        "cba_negative_cny_basis": -col["cba_cny_basis_bps"],
        "cba_consensus_basis": col["cba_consensus_basis_bps"],
        "cba_negative_consensus_basis": -col["cba_consensus_basis_bps"],
        "cba_inverse_rub_momentum_1": -col["cba_rub_quote_ret_1"],
        "cba_inverse_rub_momentum_2": -col["cba_rub_quote_ret_2"],
        "cba_inverse_rub_momentum_5": -col["cba_rub_quote_ret_5"],
        "cba_direct_usd_disagreement": col["cba_direct_minus_usd_bps"],
        "cba_direct_cny_disagreement": col["cba_direct_minus_cny_bps"],
        "cba_usd_cny_disagreement": col["cba_usd_minus_cny_bps"],
        "cba_consensus_stale20": delayed_by_currency(
            col["cba_consensus_basis_bps"][:, None], index, rows=20,
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
    screen_rows = []
    for candidate in SCREEN_ORDER:
        item = _evaluate(outputs[candidate], (2024,), POLICY, y, benefit, dates, currencies)
        item.update({"candidate": candidate, "period": "screen_2024", **POLICY})
        screen_rows.append(item)
    chosen = _choose(pd.DataFrame(screen_rows))
    primary = _load(PRIMARY)["logit50_extra50"]
    geometry = _load(GEOMETRY)["primary75_geometry_min75_max2525"]
    for base_name, base in (("primary", primary), ("geometry", geometry)):
        for weight in (.10, .25):
            name = f"{base_name}{int((1-weight)*100)}_{chosen}{int(weight*100)}"
            outputs[name] = combine_causal(
                [base, outputs[chosen]], (1.0 - weight, weight), dates, currencies,
            )
    base_score = row_scores(primary, len(y))
    local_score = raw[chosen]
    overlay = base_score.copy()
    overlay[currencies == "AMD"] = local_score[currencies == "AMD"]
    outputs["primary_amd_local_overlay"] = _outputs(overlay, y, dates)
    order = SCREEN_ORDER + tuple(
        f"{base_name}{int((1-weight)*100)}_{chosen}{int(weight*100)}"
        for base_name in ("primary", "geometry") for weight in (.10, .25)
    ) + ("primary_amd_local_overlay",)
    rows = []
    for candidate in order:
        for period, years in (
            ("screen_2024", (2024,)), ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)), ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(outputs[candidate], years, POLICY, y, benefit, dates, currencies)
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
    screen = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(screen, outputs, (2025, 2026), y, benefit, dates, currencies)
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "armenian_central_bank_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in order:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CF", "variants": order, "fixed_policy": POLICY,
        "source": "Central Bank of Armenia official SOAP range service",
        "selection_period": 2024, "selected_formula": chosen,
        "payload_sha256": digest,
        "asof_rule": "CBA effective date strictly before signal date; CBR date <= signal date",
        "publication_time_assumed": False, "physical_future_corruption_check": True,
        "stale_control_rows_per_currency": 20,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    display = results.sort_values(["period", "predeclared_order"])
    print(f"Selected on 2024: {chosen}\n")
    print(display[[
        "candidate", "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min", "robustness",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
