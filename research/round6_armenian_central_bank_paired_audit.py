"""Packet-CH: paired audit of the Armenian-central-bank consensus signal."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import B, _bootstrap_all
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_shock_weight_audit import circular_max_difference
from research.round6_resolved_models import _fire


OUT = Path("results/research/round6/armenian_central_bank_paired_audit")
SOURCE = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
GEOMETRY = Path("results/research/round6/cny_expert_geometry/outputs.pkl")
REGIME = Path("results/research/round6/cny_error_regime/outputs.pkl")
COMPARISONS = (
    ("cba_freshness", "cba_consensus_stale20", "cba_consensus_basis"),
    ("cba_on_primary", "logit50_extra50", "primary75_cba_consensus_basis25"),
    (
        "cba_on_label_free_geometry",
        "primary75_geometry_min75_max2525",
        "geometry75_cba_consensus_basis25",
    ),
    (
        "cba_geometry_vs_regime",
        "primary75_regime_logit25",
        "geometry75_cba_consensus_basis25",
    ),
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _holm(values):
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[position])))
        result[position] = running
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    outputs = _load(SOURCE)
    outputs["logit50_extra50"] = _load(PRIMARY)["logit50_extra50"]
    outputs["primary75_geometry_min75_max2525"] = _load(GEOMETRY)[
        "primary75_geometry_min75_max2525"
    ]
    outputs["primary75_regime_logit25"] = _load(REGIME)["primary75_regime_logit25"]
    rows, overlaps = [], []
    for hypothesis, control, challenger in COMPARISONS:
        valid_c, fired_c = _fire(outputs[control], (2025, 2026), POLICY, y, dates, currencies)
        valid_h, fired_h = _fire(outputs[challenger], (2025, 2026), POLICY, y, dates, currencies)
        if not np.array_equal(valid_c, valid_h):
            raise AssertionError(f"comparison rows differ for {hypothesis}")
        valid = valid_c
        draws = _bootstrap_all(y, benefit, dates, valid, {
            control: fired_c, challenger: fired_h,
        })
        ld = draws[challenger]["lift"] - draws[control]["lift"]
        bd = draws[challenger]["benefit"] - draws[control]["benefit"]
        ld = ld[np.isfinite(ld)]
        bd = bd[np.isfinite(bd)]
        control_lift = float(y[valid & fired_c].mean() / y[valid].mean())
        challenger_lift = float(y[valid & fired_h].mean() / y[valid].mean())
        control_benefit = float(np.nanmean(benefit[valid & fired_c]))
        challenger_benefit = float(np.nanmean(benefit[valid & fired_h]))
        circular = circular_max_difference(
            y, dates, currencies, valid, {control: fired_c, challenger: fired_h}, control,
        ).iloc[0]
        rows.append({
            "hypothesis": hypothesis, "control": control, "challenger": challenger,
            "control_lift": control_lift, "challenger_lift": challenger_lift,
            "lift_difference": challenger_lift - control_lift,
            "lift_difference_ci_low": float(np.quantile(ld, .025)),
            "lift_difference_ci_high": float(np.quantile(ld, .975)),
            "p_challenger_not_better": float((np.sum(ld <= 0) + 1) / (len(ld) + 1)),
            "control_benefit_bps": control_benefit,
            "challenger_benefit_bps": challenger_benefit,
            "benefit_difference_bps": challenger_benefit - control_benefit,
            "benefit_difference_ci_low": float(np.quantile(bd, .025)),
            "benefit_difference_ci_high": float(np.quantile(bd, .975)),
            "circular_p_unadjusted": float(circular.circular_p_unadjusted),
        })
        for subset, mask in {
            "control": fired_c, "challenger": fired_h,
            "intersection": fired_c & fired_h,
            "control_only": fired_c & ~fired_h,
            "challenger_only": fired_h & ~fired_c,
            "neither": ~fired_c & ~fired_h,
        }.items():
            active = valid & mask
            overlaps.append({
                "hypothesis": hypothesis, "subset": subset, "n": int(active.sum()),
                "share": float(active.sum() / valid.sum()),
                "lift": float(y[active].mean() / y[valid].mean()) if active.any() else np.nan,
                "benefit_bps": float(np.nanmean(benefit[active])) if active.any() else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary["p_holm_four_hypotheses"] = _holm(summary.p_challenger_not_better.to_numpy())
    summary["lift_increment_supported"] = (
        summary.lift_difference_ci_low.gt(0)
        & summary.p_holm_four_hypotheses.lt(.05)
    )
    summary["joint_lift_benefit_supported"] = (
        summary.lift_increment_supported & summary.benefit_difference_ci_low.gt(0)
    )
    summary.to_csv(OUT / "paired_audit.csv", index=False)
    pd.DataFrame(overlaps).to_csv(OUT / "signal_overlap.csv", index=False)
    multi = pd.read_csv("results/research/round6/multihorizon_case_audit/candidate_summary.csv")
    multi = multi[multi.candidate == "geometry_cba_blend"].iloc[0]
    gates = {
        "all_horizons_case_lift_at_least_1_30": bool(multi.all_horizons_lift_1p30),
        "all_horizons_symmetric_benefit_positive": bool(
            multi.all_horizons_symmetric_benefit_positive
        ),
        "all_horizons_future_benefit_positive": bool(
            multi.all_horizons_future_benefit_positive
        ),
        "all_horizon_frequencies_between_1_and_2": bool(
            multi.frequency_min >= 1 and multi.frequency_max <= 2
        ),
    }
    gates["all_case_gates"] = bool(all(gates.values()))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CH", "comparisons": COMPARISONS, "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "Holm adjustment across four predeclared comparisons",
        "case_gates": gates, "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nCASE GATES\n" + json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
