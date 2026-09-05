"""Packet-CE: paired audit of the label-free two-expert geometry."""
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


OUT = Path("results/research/round6/cny_expert_geometry_paired_audit")
SOURCE = Path("results/research/round6/cny_expert_geometry/outputs.pkl")
RESULTS = Path("results/research/round6/cny_expert_geometry/matched_results.csv")
BOOTSTRAP = Path("results/research/round6/cny_expert_geometry/block_bootstrap_2025_2026.csv")
CIRCULAR = Path("results/research/round6/cny_expert_geometry/circular_shift_multiplicity.csv")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
ROUTER = Path("results/research/round6/cny_expert_router/outputs.pkl")
REGIME = Path("results/research/round6/cny_error_regime/outputs.pkl")
BLEND = "primary75_geometry_min75_max2525"
COMPARISONS = (
    ("geometry_freshness", "geometry_stale20_minimum", "geometry_minimum"),
    ("geometry_vs_equal", "shadow50_survival50", "geometry_min75_max25"),
    ("geometry_vs_router", "router_tree_hard", "geometry_min75_max25"),
    ("blend_vs_primary", "logit50_extra50", BLEND),
    ("blend_vs_regime", "primary75_regime_logit25", BLEND),
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
    router = _load(ROUTER)
    outputs["router_tree_hard"] = router["router_tree_hard"]
    outputs["shadow50_survival50"] = router["shadow50_survival50"]
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
        lift_diff = draws[challenger]["lift"] - draws[control]["lift"]
        benefit_diff = draws[challenger]["benefit"] - draws[control]["benefit"]
        lift_diff = lift_diff[np.isfinite(lift_diff)]
        benefit_diff = benefit_diff[np.isfinite(benefit_diff)]
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
            "lift_difference_ci_low": float(np.quantile(lift_diff, .025)),
            "lift_difference_ci_high": float(np.quantile(lift_diff, .975)),
            "p_challenger_not_better": float((np.sum(lift_diff <= 0) + 1) / (len(lift_diff) + 1)),
            "control_benefit_bps": control_benefit,
            "challenger_benefit_bps": challenger_benefit,
            "benefit_difference_bps": challenger_benefit - control_benefit,
            "benefit_difference_ci_low": float(np.quantile(benefit_diff, .025)),
            "benefit_difference_ci_high": float(np.quantile(benefit_diff, .975)),
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
                "hypothesis": hypothesis, "subset": subset,
                "n": int(active.sum()), "share": float(active.sum() / valid.sum()),
                "lift": float(y[active].mean() / y[valid].mean()) if active.any() else np.nan,
                "benefit_bps": float(np.nanmean(benefit[active])) if active.any() else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary["p_holm_five_hypotheses"] = _holm(summary.p_challenger_not_better.to_numpy())
    summary["lift_increment_supported"] = (
        summary.lift_difference_ci_low.gt(0)
        & summary.p_holm_five_hypotheses.lt(.05)
    )
    summary["joint_lift_benefit_supported"] = (
        summary.lift_increment_supported & summary.benefit_difference_ci_low.gt(0)
    )
    summary.to_csv(OUT / "paired_audit.csv", index=False)
    pd.DataFrame(overlaps).to_csv(OUT / "signal_overlap.csv", index=False)

    metrics = pd.read_csv(RESULTS)
    yearly = metrics[
        (metrics.candidate == BLEND)
        & metrics.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    combined = metrics[
        (metrics.candidate == BLEND) & (metrics.period == "combined_2025_2026")
    ].iloc[0]
    boot = pd.read_csv(BOOTSTRAP)
    boot = boot[boot.candidate == BLEND].iloc[0]
    circ = pd.read_csv(CIRCULAR)
    circ = circ[circ.policy == BLEND].iloc[0]
    pair = summary[summary.hypothesis == "blend_vs_primary"].iloc[0]
    gates = {
        "annual_lift_at_least_1_30": bool(yearly.lift.ge(1.30).all()),
        "annual_rate_between_1_and_2": bool(yearly.frequency.between(1, 2).all()),
        "combined_rate_between_1_and_2": bool(1 <= combined.frequency <= 2),
        "minimum_currency_lift_at_least_1_30": bool(combined.corridor_lift_min >= 1.30),
        "standalone_lift_ci_above_1": bool(boot.lift_ci_low > 1.0),
        "circular_max_adjusted_p_below_005": bool(circ.circular_shift_p_max_adjusted < .05),
        "paired_lift_ci_above_primary": bool(pair.lift_difference_ci_low > 0.0),
        "paired_benefit_ci_above_primary": bool(pair.benefit_difference_ci_low > 0.0),
    }
    gates["all_gates"] = bool(all(gates.values()))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CE", "comparisons": COMPARISONS, "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "Holm adjustment across five predeclared comparisons",
        "operational_gates": gates,
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nOPERATIONAL GATES\n" + json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
