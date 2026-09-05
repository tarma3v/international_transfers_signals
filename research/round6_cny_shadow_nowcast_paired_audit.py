"""Packet-BR paired audit of the causal CNY shadow-rate nowcast."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import B, _bootstrap_all
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_shock_weight_audit import circular_max_difference
from research.round6_resolved_models import _evaluate, _fire


OUT = Path("results/research/round6/cny_shadow_nowcast_paired_audit")
SOURCE = Path("results/research/round6/cny_shadow_nowcast/outputs.pkl")
RESULTS = Path("results/research/round6/cny_shadow_nowcast/matched_results.csv")
BOOTSTRAP = Path("results/research/round6/cny_shadow_nowcast/block_bootstrap_2025_2026.csv")
CIRCULAR = Path("results/research/round6/cny_shadow_nowcast/circular_shift_multiplicity.csv")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
PRIMARY_RESULTS = Path("results/research/round6/cny_consensus/matched_results.csv")
COMPARISONS = (
    ("shadow_freshness", "shadow_close_stale20", "shadow_close_basis"),
    ("cross_correction", "shadow_close_basis", "shadow_close_cross5"),
    ("primary_increment", "logit50_extra50", "primary75_shadow25"),
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
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
    outputs["logit50_extra50"] = _load(PRIMARY)["logit50_extra50"]
    raw = np.full(len(y), np.nan)
    for year, item in outputs["shadow_close_basis"].items():
        raw[item["calib_idx"]] = item["calib_score"]
        raw[item["test_idx"]] = item["test_score"]
    stale_raw = delayed_by_currency(raw[:, None], index, rows=20)[:, 0]
    outputs["shadow_close_stale20"] = _outputs(stale_raw, y, dates)

    stale_metrics = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        item = _evaluate(
            outputs["shadow_close_stale20"], years, POLICY,
            y, benefit, dates, currencies,
        )
        item.update({"candidate": "shadow_close_stale20", "period": period})
        stale_metrics.append(item)
    pd.DataFrame(stale_metrics).to_csv(OUT / "stale_control_metrics.csv", index=False)

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
        benefit_difference = (
            draws[challenger]["benefit"] - draws[control]["benefit"]
        )
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
    primary_metrics = pd.read_csv(PRIMARY_RESULTS)
    bootstrap = pd.read_csv(BOOTSTRAP)
    circular = pd.read_csv(CIRCULAR)
    direct_yearly = metrics[
        (metrics.candidate == "shadow_close_basis")
        & metrics.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    direct_combined = metrics[
        (metrics.candidate == "shadow_close_basis")
        & (metrics.period == "combined_2025_2026")
    ].iloc[0]
    direct_boot = bootstrap[bootstrap.candidate == "shadow_close_basis"].iloc[0]
    direct_circular = circular[circular.policy == "shadow_close_basis"].iloc[0]
    freshness = summary[summary.hypothesis == "shadow_freshness"].iloc[0]
    direct_gates = {
        "annual_lift_at_least_1_30": bool(direct_yearly.lift.ge(1.30).all()),
        "annual_rate_between_1_and_2": bool(direct_yearly.frequency.between(1, 2).all()),
        "minimum_currency_lift_at_least_1_30": bool(
            direct_combined.corridor_lift_min >= 1.30
        ),
        "standalone_lift_ci_above_1": bool(direct_boot.lift_ci_low > 1.0),
        "circular_max_adjusted_p_below_005": bool(
            direct_circular.circular_shift_p_max_adjusted < .05
        ),
        "fresh_vs_stale_paired_lift_ci_above_0": bool(
            freshness.lift_difference_ci_low > 0
        ),
    }
    direct_gates["all_direct_feasibility_gates"] = bool(all(direct_gates.values()))

    blend_yearly = metrics[
        (metrics.candidate == "primary75_shadow25")
        & metrics.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    blend_combined = metrics[
        (metrics.candidate == "primary75_shadow25")
        & (metrics.period == "combined_2025_2026")
    ].iloc[0]
    primary_combined = primary_metrics[
        (primary_metrics.candidate == "logit50_extra50")
        & (primary_metrics.period == "combined_2025_2026")
    ].iloc[0]
    increment = summary[summary.hypothesis == "primary_increment"].iloc[0]
    blend_gates = {
        "annual_lift_at_least_1_30": bool(blend_yearly.lift.ge(1.30).all()),
        "annual_rate_between_1_and_2": bool(blend_yearly.frequency.between(1, 2).all()),
        "minimum_currency_lift_at_least_1_30": bool(
            blend_combined.corridor_lift_min >= 1.30
        ),
        "minimum_quarter_rate_at_least_0_95": bool(
            blend_combined.quarter_frequency_min >= .95
        ),
        "combined_lift_above_primary": bool(blend_combined.lift > primary_combined.lift),
        "combined_benefit_above_primary": bool(
            blend_combined.forward_benefit_bps > primary_combined.forward_benefit_bps
        ),
        "paired_lift_ci_excludes_zero": bool(increment.lift_difference_ci_low > 0),
        "paired_benefit_ci_excludes_zero": bool(increment.benefit_difference_ci_low > 0),
        "holm_p_below_005": bool(increment.p_holm_three_hypotheses < .05),
    }
    blend_gates["all_blend_promotion_gates"] = bool(all(blend_gates.values()))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BR",
        "comparisons": COMPARISONS,
        "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "Holm adjustment across three predeclared hypotheses",
        "stale_control_rows_per_currency": 20,
        "direct_feasibility_gates": direct_gates,
        "blend_promotion_gates": blend_gates,
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nDIRECT GATES\n" + json.dumps(direct_gates, indent=2))
    print("\nBLEND GATES\n" + json.dumps(blend_gates, indent=2))


if __name__ == "__main__":
    main()
