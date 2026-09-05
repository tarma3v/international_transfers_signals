"""Packet-BU paired audit of the causal survival decomposition."""
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


OUT = Path("results/research/round6/cny_survival_hazard_paired_audit")
SOURCE = Path("results/research/round6/cny_survival_hazard/outputs.pkl")
RESULTS = Path("results/research/round6/cny_survival_hazard/matched_results.csv")
BOOTSTRAP = Path("results/research/round6/cny_survival_hazard/block_bootstrap_2025_2026.csv")
CIRCULAR = Path("results/research/round6/cny_survival_hazard/circular_shift_multiplicity.csv")
SHADOW = Path("results/research/round6/cny_shadow_nowcast/outputs.pkl")
COMPARISONS = (
    ("hazard_freshness", "survival_hazard_stale20", "survival_hazard_product"),
    ("nested_target_increment", "survival_direct_h5", "survival_cumulative_geometric"),
    ("shadow_increment", "shadow_close_basis", "survival_cumulative_geometric"),
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values[position]))
        running = max(running, value)
        adjusted[position] = running
    return adjusted


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
    outputs = _load(SOURCE)
    outputs["shadow_close_basis"] = _load(SHADOW)["shadow_close_basis"]
    rows, overlap_rows = [], []
    for hypothesis, control, challenger in COMPARISONS:
        valid_control, fired_control = _fire(
            outputs[control], (2025, 2026), POLICY, y, dates, currencies,
        )
        valid_challenger, fired_challenger = _fire(
            outputs[challenger], (2025, 2026), POLICY, y, dates, currencies,
        )
        if not np.array_equal(valid_control, valid_challenger):
            raise AssertionError(f"comparison rows differ for {hypothesis}")
        valid = valid_control
        masks = {control: fired_control, challenger: fired_challenger}
        draws = _bootstrap_all(y, benefit, dates, valid, masks)
        lift_difference = draws[challenger]["lift"] - draws[control]["lift"]
        benefit_difference = draws[challenger]["benefit"] - draws[control]["benefit"]
        finite_lift = lift_difference[np.isfinite(lift_difference)]
        finite_benefit = benefit_difference[np.isfinite(benefit_difference)]
        control_lift = float(y[valid & fired_control].mean() / y[valid].mean())
        challenger_lift = float(y[valid & fired_challenger].mean() / y[valid].mean())
        control_benefit = float(np.nanmean(benefit[valid & fired_control]))
        challenger_benefit = float(np.nanmean(benefit[valid & fired_challenger]))
        circular = circular_max_difference(
            y, dates, currencies, valid, masks, control,
        ).iloc[0]
        rows.append({
            "hypothesis": hypothesis,
            "control": control,
            "challenger": challenger,
            "control_lift": control_lift,
            "challenger_lift": challenger_lift,
            "lift_difference": challenger_lift - control_lift,
            "lift_difference_ci_low": float(np.quantile(finite_lift, .025)),
            "lift_difference_ci_high": float(np.quantile(finite_lift, .975)),
            "p_challenger_not_better": float(
                (np.sum(finite_lift <= 0) + 1) / (len(finite_lift) + 1)
            ),
            "control_benefit_bps": control_benefit,
            "challenger_benefit_bps": challenger_benefit,
            "benefit_difference_bps": challenger_benefit - control_benefit,
            "benefit_difference_ci_low": float(np.quantile(finite_benefit, .025)),
            "benefit_difference_ci_high": float(np.quantile(finite_benefit, .975)),
            "circular_p_unadjusted": float(circular.circular_p_unadjusted),
        })
        subsets = {
            "control": fired_control,
            "challenger": fired_challenger,
            "intersection": fired_control & fired_challenger,
            "control_only": fired_control & ~fired_challenger,
            "challenger_only": fired_challenger & ~fired_control,
            "neither": ~fired_control & ~fired_challenger,
        }
        for subset, mask in subsets.items():
            active = valid & mask
            values = benefit[active & np.isfinite(benefit)]
            overlap_rows.append({
                "hypothesis": hypothesis,
                "subset": subset,
                "n": int(active.sum()),
                "share": float(active.sum() / valid.sum()),
                "target_rate": float(y[active].mean()) if active.any() else np.nan,
                "lift": (
                    float(y[active].mean() / y[valid].mean())
                    if active.any() else np.nan
                ),
                "benefit_bps": float(values.mean()) if len(values) else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary["p_holm_three_hypotheses"] = _holm_adjust(
        summary.p_challenger_not_better.to_numpy()
    )
    summary.to_csv(OUT / "paired_audit.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "signal_overlap.csv", index=False)

    metrics = pd.read_csv(RESULTS)
    yearly = metrics[
        (metrics.candidate == "survival_cumulative_geometric")
        & metrics.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    combined = metrics[
        (metrics.candidate == "survival_cumulative_geometric")
        & (metrics.period == "combined_2025_2026")
    ].iloc[0]
    bootstrap = pd.read_csv(BOOTSTRAP)
    boot = bootstrap[
        bootstrap.candidate == "survival_cumulative_geometric"
    ].iloc[0]
    circular = pd.read_csv(CIRCULAR)
    circ = circular[
        circular.policy == "survival_cumulative_geometric"
    ].iloc[0]
    gates = {
        "annual_lift_at_least_1_30": bool(yearly.lift.ge(1.30).all()),
        "annual_rate_between_1_and_2": bool(yearly.frequency.between(1, 2).all()),
        "minimum_currency_lift_at_least_1_30": bool(combined.corridor_lift_min >= 1.30),
        "minimum_quarter_rate_at_least_1": bool(combined.quarter_frequency_min >= 1.0),
        "standalone_lift_ci_above_1": bool(boot.lift_ci_low > 1.0),
        "circular_max_adjusted_p_below_005": bool(
            circ.circular_shift_p_max_adjusted < .05
        ),
    }
    gates["all_operational_gates"] = bool(all(gates.values()))
    increments = {}
    for hypothesis in ("nested_target_increment", "shadow_increment"):
        row = summary[summary.hypothesis == hypothesis].iloc[0]
        increments[hypothesis] = {
            "paired_lift_ci_above_zero": bool(row.lift_difference_ci_low > 0),
            "paired_benefit_ci_above_zero": bool(row.benefit_difference_ci_low > 0),
            "holm_p_below_005": bool(row.p_holm_three_hypotheses < .05),
        }
    freshness = summary[summary.hypothesis == "hazard_freshness"].iloc[0]
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BU",
        "comparisons": COMPARISONS,
        "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "Holm adjustment across three predeclared hypotheses",
        "hazard_freshness_supported": bool(freshness.lift_difference_ci_low > 0),
        "operational_gates": gates,
        "incremental_claims": increments,
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nOPERATIONAL GATES\n" + json.dumps(gates, indent=2))
    print("\nINCREMENTS\n" + json.dumps(increments, indent=2))


if __name__ == "__main__":
    main()
