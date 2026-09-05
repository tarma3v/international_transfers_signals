"""Packet-BL paired audit of full-lifecycle CNY path challengers."""
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
from research.round6_cny_lifecycle import YEARS
from research.round6_cny_shock_weight_audit import circular_max_difference
from research.round6_resolved_models import _fire


OUT = Path("results/research/round6/cny_rocket_lifecycle_paired_audit")
SOURCE = Path("results/research/round6/cny_rocket_lifecycle/outputs.pkl")
RESULTS = Path("results/research/round6/cny_rocket_lifecycle/matched_results.csv")
CONTROL = "primary_resolved2000"
CHALLENGERS = (
    "primary75_rocket25_lifecycle",
    "primary_then_regime2024",
)


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
    masks, valid = {}, None
    for candidate in (CONTROL, *CHALLENGERS):
        current_valid, fired = _fire(
            outputs[candidate], YEARS, POLICY, y, dates, currencies,
        )
        if valid is None:
            valid = current_valid
        elif not np.array_equal(valid, current_valid):
            raise AssertionError("lifecycle comparison rows differ")
        masks[candidate] = fired
    draws = _bootstrap_all(y, benefit, dates, valid, masks)
    circular = circular_max_difference(
        y, dates, currencies, valid, masks, CONTROL,
    )
    rows, overlap_rows = [], []
    control_lift = float(y[valid & masks[CONTROL]].mean() / y[valid].mean())
    control_benefit = float(np.nanmean(benefit[valid & masks[CONTROL]]))
    for challenger in CHALLENGERS:
        lift = float(y[valid & masks[challenger]].mean() / y[valid].mean())
        lift_difference = draws[challenger]["lift"] - draws[CONTROL]["lift"]
        benefit_difference = draws[challenger]["benefit"] - draws[CONTROL]["benefit"]
        finite_lift = lift_difference[np.isfinite(lift_difference)]
        finite_benefit = benefit_difference[np.isfinite(benefit_difference)]
        shift = circular[circular.candidate == challenger].iloc[0]
        rows.append({
            "candidate": challenger,
            "control_lift": control_lift,
            "challenger_lift": lift,
            "lift_difference": lift - control_lift,
            "lift_difference_ci_low": float(np.quantile(finite_lift, .025)),
            "lift_difference_ci_high": float(np.quantile(finite_lift, .975)),
            "p_challenger_not_better": float(
                (np.sum(finite_lift <= 0) + 1) / (len(finite_lift) + 1)
            ),
            "control_benefit_bps": control_benefit,
            "challenger_benefit_bps": float(np.nanmean(benefit[valid & masks[challenger]])),
            "benefit_difference_bps": float(
                np.nanmean(benefit[valid & masks[challenger]]) - control_benefit
            ),
            "benefit_difference_ci_low": float(np.quantile(finite_benefit, .025)),
            "benefit_difference_ci_high": float(np.quantile(finite_benefit, .975)),
            "circular_p_unadjusted": float(shift.circular_p_unadjusted),
            "circular_p_max_adjusted": float(shift.circular_p_max_adjusted),
        })
        subsets = {
            "control": masks[CONTROL],
            "challenger": masks[challenger],
            "intersection": masks[CONTROL] & masks[challenger],
            "control_only": masks[CONTROL] & ~masks[challenger],
            "challenger_only": masks[challenger] & ~masks[CONTROL],
            "neither": ~masks[CONTROL] & ~masks[challenger],
        }
        for subset, mask in subsets.items():
            active = valid & mask
            values = benefit[active & np.isfinite(benefit)]
            overlap_rows.append({
                "candidate": challenger,
                "subset": subset,
                "n": int(active.sum()),
                "share": float(active.sum() / valid.sum()),
                "target_rate": float(y[active].mean()) if active.any() else np.nan,
                "lift": float(y[active].mean() / y[valid].mean()) if active.any() else np.nan,
                "benefit_bps": float(values.mean()) if len(values) else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "paired_multiplicity_audit.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "signal_overlap.csv", index=False)

    metrics = pd.read_csv(RESULTS)
    control_years = metrics[
        (metrics.candidate == CONTROL) & metrics.period.isin(tuple(map(str, YEARS)))
    ]
    decisions = {}
    for challenger in CHALLENGERS:
        yearly = metrics[
            (metrics.candidate == challenger)
            & metrics.period.isin(tuple(map(str, YEARS)))
        ]
        combined = metrics[
            (metrics.candidate == challenger) & (metrics.period == "2017_2026")
        ].iloc[0]
        pair = summary[summary.candidate == challenger].iloc[0]
        gates = {
            "all_annual_lift_at_least_1_30": bool(yearly.lift.ge(1.30).all()),
            "all_annual_rate_between_1_and_2": bool(yearly.frequency.between(1, 2).all()),
            "minimum_currency_lift_at_least_1_30": bool(combined.corridor_lift_min >= 1.30),
            "minimum_annual_lift": float(yearly.lift.min()),
            "control_minimum_annual_lift": float(control_years.lift.min()),
            "minimum_quarter_rate": float(combined.quarter_frequency_min),
            "strict_quarter_rate_gate_0_90": bool(combined.quarter_frequency_min >= .90),
            "paired_lift_ci_excludes_zero": bool(pair.lift_difference_ci_low > 0),
            "max_adjusted_p_below_005": bool(pair.circular_p_max_adjusted < .05),
        }
        gates["statistical_lifecycle_superiority"] = bool(
            gates["paired_lift_ci_excludes_zero"]
            and gates["max_adjusted_p_below_005"]
        )
        decisions[challenger] = gates
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BL",
        "control": CONTROL,
        "challengers": CHALLENGERS,
        "years": YEARS,
        "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "max lift difference over two challengers under circular date shifts",
        "decisions": decisions,
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective lifecycle audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nDECISIONS\n" + json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
