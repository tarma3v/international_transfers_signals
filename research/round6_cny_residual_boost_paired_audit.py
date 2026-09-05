"""Packet-BN paired audit of low-dose causal residual correction."""
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


OUT = Path("results/research/round6/cny_residual_boost_paired_audit")
SOURCE = Path("results/research/round6/cny_residual_boost/outputs.pkl")
RESULTS = Path("results/research/round6/cny_residual_boost/matched_results.csv")
COMPARISONS = (
    ("residual_freshness", "residual_hist_stale20_25", "residual_hist25"),
    ("hist_primary_increment", "logit50_extra50", "residual_hist25"),
    ("extra_primary_increment", "logit50_extra50", "residual_extra25"),
)


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
    with SOURCE.open("rb") as handle:
        outputs = pickle.load(handle)
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
            "control_benefit_bps": float(np.nanmean(benefit[valid & fired_control])),
            "challenger_benefit_bps": float(np.nanmean(benefit[valid & fired_challenger])),
            "benefit_difference_bps": float(
                np.nanmean(benefit[valid & fired_challenger])
                - np.nanmean(benefit[valid & fired_control])
            ),
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
                "lift": float(y[active].mean() / y[valid].mean()) if active.any() else np.nan,
                "benefit_bps": float(values.mean()) if len(values) else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary["p_holm_three_hypotheses"] = _holm_adjust(
        summary.p_challenger_not_better.to_numpy()
    )
    summary.to_csv(OUT / "paired_audit.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "signal_overlap.csv", index=False)

    metrics = pd.read_csv(RESULTS)
    decisions = {}
    for candidate, hypothesis in (
        ("residual_hist25", "hist_primary_increment"),
        ("residual_extra25", "extra_primary_increment"),
    ):
        yearly = metrics[
            (metrics.candidate == candidate)
            & metrics.period.isin(("retrospective_2025", "retrospective_2026"))
        ]
        combined = metrics[
            (metrics.candidate == candidate)
            & (metrics.period == "combined_2025_2026")
        ].iloc[0]
        control = metrics[
            (metrics.candidate == "logit50_extra50")
            & (metrics.period == "combined_2025_2026")
        ].iloc[0]
        pair = summary[summary.hypothesis == hypothesis].iloc[0]
        gates = {
            "annual_lift_at_least_1_30": bool(yearly.lift.ge(1.30).all()),
            "annual_rate_between_1_and_2": bool(yearly.frequency.between(1, 2).all()),
            "minimum_currency_lift_at_least_1_30": bool(combined.corridor_lift_min >= 1.30),
            "minimum_quarter_rate_at_least_1": bool(combined.quarter_frequency_min >= 1.0),
            "combined_lift_above_primary": bool(combined.lift > control.lift),
            "paired_lift_ci_excludes_zero": bool(pair.lift_difference_ci_low > 0),
            "holm_p_below_005": bool(pair.p_holm_three_hypotheses < .05),
        }
        gates["all_promotion_gates"] = bool(all(gates.values()))
        decisions[candidate] = gates
    fresh_pair = summary[summary.hypothesis == "residual_freshness"].iloc[0]
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BN",
        "comparisons": COMPARISONS,
        "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "Holm adjustment across three predeclared paired hypotheses",
        "circular_shift": "separate fixed-comparison calendar-alignment diagnostic",
        "fresh_hist_supported": bool(fresh_pair.lift_difference_ci_low > 0),
        "promotion_gates": decisions,
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nPROMOTION GATES\n" + json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
