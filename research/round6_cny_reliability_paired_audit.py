"""Packet-BC paired and multiplicity audit for causal reliability surfaces."""
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


OUT = Path("results/research/round6/cny_reliability_paired_audit")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
PRIMARY_RESULTS = Path("results/research/round6/cny_consensus/matched_results.csv")
SOURCE = Path("results/research/round6/cny_reliability_surface/outputs.pkl")
SOURCE_RESULTS = Path("results/research/round6/cny_reliability_surface/matched_results.csv")
CONTROL = "logit50_extra50"
CHALLENGERS = (
    "pooled_hit_lcb",
    "shrunk_hit_lcb",
    "shrunk_benefit_lcb",
    "reliability_benefit_equal",
    "primary75_reliability25",
)
PERIODS = {
    "screen_2024": (2024,),
    "retrospective_2025": (2025,),
    "retrospective_2026": (2026,),
    "combined_2025_2026": (2025, 2026),
}


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
    primary = _load(PRIMARY)[CONTROL]
    source = _load(SOURCE)
    outputs = {CONTROL: primary, **{name: source[name] for name in CHALLENGERS}}
    frames, overlap_rows = [], []
    for period, years in PERIODS.items():
        masks, common_valid = {}, None
        for name, output in outputs.items():
            valid, fired = _fire(output, years, POLICY, y, dates, currencies)
            if common_valid is None:
                common_valid = valid
            elif not np.array_equal(common_valid, valid):
                raise AssertionError(f"comparison rows differ in {period}")
            masks[name] = fired
        draws = _bootstrap_all(y, benefit, dates, common_valid, masks)
        control_lift = float(y[common_valid & masks[CONTROL]].mean() / y[common_valid].mean())
        control_benefit = float(np.nanmean(benefit[common_valid & masks[CONTROL]]))
        rows = []
        for name in CHALLENGERS:
            fired = masks[name]
            lift = float(y[common_valid & fired].mean() / y[common_valid].mean())
            lift_difference = draws[name]["lift"] - draws[CONTROL]["lift"]
            benefit_difference = draws[name]["benefit"] - draws[CONTROL]["benefit"]
            finite_lift = lift_difference[np.isfinite(lift_difference)]
            finite_benefit = benefit_difference[np.isfinite(benefit_difference)]
            rows.append({
                "period": period,
                "candidate": name,
                "control_lift": control_lift,
                "challenger_lift": lift,
                "lift_difference": lift - control_lift,
                "lift_difference_ci_low": float(np.quantile(finite_lift, .025)),
                "lift_difference_ci_high": float(np.quantile(finite_lift, .975)),
                "p_challenger_not_better": float(
                    (np.sum(finite_lift <= 0) + 1) / (len(finite_lift) + 1)
                ),
                "control_benefit_bps": control_benefit,
                "challenger_benefit_bps": float(np.nanmean(benefit[common_valid & fired])),
                "benefit_difference_ci_low": float(np.quantile(finite_benefit, .025)),
                "benefit_difference_ci_high": float(np.quantile(finite_benefit, .975)),
            })
        paired = pd.DataFrame(rows)
        circular = circular_max_difference(
            y, dates, currencies, common_valid, masks, CONTROL,
        )
        frames.append(paired.merge(circular, on="candidate", how="left"))
        if period == "combined_2025_2026":
            for candidate in ("pooled_hit_lcb", "shrunk_hit_lcb"):
                subsets = {
                    "control": masks[CONTROL],
                    "challenger": masks[candidate],
                    "intersection": masks[CONTROL] & masks[candidate],
                    "control_only": masks[CONTROL] & ~masks[candidate],
                    "challenger_only": masks[candidate] & ~masks[CONTROL],
                    "neither": ~masks[CONTROL] & ~masks[candidate],
                }
                for subset, mask in subsets.items():
                    active = common_valid & mask
                    values = benefit[active & np.isfinite(benefit)]
                    overlap_rows.append({
                        "candidate": candidate,
                        "subset": subset,
                        "n": int(active.sum()),
                        "target_rate": float(y[active].mean()) if active.any() else np.nan,
                        "lift": (
                            float(y[active].mean() / y[common_valid].mean())
                            if active.any() else np.nan
                        ),
                        "benefit_bps": float(values.mean()) if len(values) else np.nan,
                    })
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(OUT / "paired_multiplicity_audit.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "signal_overlap.csv", index=False)

    source_metrics = pd.read_csv(SOURCE_RESULTS)
    primary_metrics = pd.read_csv(PRIMARY_RESULTS)
    control_year = primary_metrics[
        (primary_metrics.candidate == CONTROL)
        & primary_metrics.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    control_min_annual = float(control_year.lift.min())
    decisions = {}
    for candidate in CHALLENGERS:
        yearly = source_metrics[
            (source_metrics.candidate == candidate)
            & source_metrics.period.isin(("retrospective_2025", "retrospective_2026"))
        ]
        combined = source_metrics[
            (source_metrics.candidate == candidate)
            & (source_metrics.period == "combined_2025_2026")
        ].iloc[0]
        pair = summary[
            (summary.period == "combined_2025_2026")
            & (summary.candidate == candidate)
        ].iloc[0]
        gates = {
            "annual_lift_at_least_1_30": bool(yearly.lift.ge(1.30).all()),
            "annual_rate_between_1_and_2": bool(yearly.frequency.between(1, 2).all()),
            "minimum_annual_lift_not_below_primary": bool(
                yearly.lift.min() >= control_min_annual
            ),
            "minimum_currency_lift_at_least_1_30": bool(
                combined.corridor_lift_min >= 1.30
            ),
            "minimum_quarter_rate_at_least_1": bool(
                combined.quarter_frequency_min >= 1.0
            ),
            "paired_lift_ci_excludes_zero": bool(pair.lift_difference_ci_low > 0),
            "max_adjusted_p_below_005": bool(pair.circular_p_max_adjusted < .05),
        }
        decisions[candidate] = {**gates, "all_promotion_gates": bool(all(gates.values()))}
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BC",
        "control": CONTROL,
        "challengers": CHALLENGERS,
        "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "multiplicity": "max lift difference over five BB variants under circular date shifts",
        "promotion_gates": decisions,
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.sort_values(["period", "lift_difference"], ascending=[True, False]).to_string(index=False))
    print("\nPROMOTION GATES\n" + json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
