"""Packet-CG: case-definition audit over all five required horizons.

The original research optimized h=5 and reported a deliberately stricter
future-only benefit.  The case owner's Q&A additionally asks for every
horizon, defines the random-day baseline within corridor and period, and uses
the symmetric +/-h benefit.  This audit reports both definitions side by side
without refitting or selecting a model on 2025-2026.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import HORIZONS, benefit_forward_only, build_targets
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_resolved_models import _fire


OUT = Path("results/research/round6/multihorizon_case_audit")
SOURCES = {
    "primary": (
        Path("results/research/round6/cny_consensus/outputs.pkl"),
        "logit50_extra50",
    ),
    "regime_blend": (
        Path("results/research/round6/cny_error_regime/outputs.pkl"),
        "primary75_regime_logit25",
    ),
    "label_free_geometry_blend": (
        Path("results/research/round6/cny_expert_geometry/outputs.pkl"),
        "primary75_geometry_min75_max2525",
    ),
    "geometry_cba_blend": (
        Path("results/research/round6/armenian_central_bank_models/outputs.pkl"),
        "geometry75_cba_consensus_basis25",
    ),
    "resolved_error_router": (
        Path("results/research/round6/cny_expert_router/outputs.pkl"),
        "router_tree_hard",
    ),
}


def _load(path, name):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def corridor_period_adjusted_lift(y, valid, fired, currencies, dates, years):
    """Observed hit rate divided by the matched random-day expectation."""
    active = valid & fired
    if not active.any():
        return np.nan, np.nan, np.nan
    expected_hits = 0.0
    macro_lifts = []
    for currency in CORRIDORS:
        for year in years:
            period = valid & (currencies == currency) & np.asarray([
                day.year == year for day in dates
            ])
            chosen = active & (currencies == currency) & np.asarray([
                day.year == year for day in dates
            ])
            if not period.any() or not chosen.any():
                continue
            base = float(y[period].mean())
            expected_hits += float(chosen.sum()) * base
            if base > 0:
                macro_lifts.append(float(y[chosen].mean() / base))
    matched_base = expected_hits / float(active.sum())
    hit = float(y[active].mean())
    return hit / matched_base, matched_base, float(np.mean(macro_lifts))


def _benefit_forward(series, index, h):
    result = np.full(len(index), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, position, h)
        if value is not None:
            result[row] = value
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    outputs = {name: _load(*source) for name, source in SOURCES.items()}
    rows = []
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        forward = _benefit_forward(series, index, h)
        symmetric = targets[f"benefit_h{h}"]
        for candidate, output in outputs.items():
            for period_name, years in (
                ("retrospective_2025", (2025,)),
                ("retrospective_2026", (2026,)),
                ("combined_2025_2026", (2025, 2026)),
            ):
                valid, fired = _fire(output, years, POLICY, y, dates, currencies)
                active = valid & fired
                pooled_base = float(y[valid].mean())
                hit = float(y[active].mean()) if active.any() else np.nan
                adjusted, matched_base, macro = corridor_period_adjusted_lift(
                    y, valid, fired, currencies, dates, years,
                )
                corridor_frequency = []
                corridor_lifts = []
                for currency in CORRIDORS:
                    scope = valid & (currencies == currency)
                    chosen = active & (currencies == currency)
                    weeks = (max(dates[scope]) - min(dates[scope])).days / 7.0
                    corridor_frequency.append(float(chosen.sum() / weeks))
                    corridor_lifts.append(
                        float(y[chosen].mean() / y[scope].mean()) if chosen.any() else np.nan
                    )
                total_weeks = (max(dates[valid]) - min(dates[valid])).days / 7.0
                rows.append({
                    "candidate": candidate, "period": period_name, "horizon": h,
                    "n_scope": int(valid.sum()), "n_signals": int(active.sum()),
                    "frequency": float(active.sum() / len(CORRIDORS) / total_weeks),
                    "corridor_frequency_min": float(np.nanmin(corridor_frequency)),
                    "corridor_frequency_max": float(np.nanmax(corridor_frequency)),
                    "hit_rate": hit, "pooled_random_day_rate": pooled_base,
                    "pooled_lift": hit / pooled_base,
                    "matched_random_day_rate": matched_base,
                    "case_lift": adjusted, "macro_corridor_year_lift": macro,
                    "corridor_lift_min": float(np.nanmin(corridor_lifts)),
                    "symmetric_benefit_bps": float(np.nanmean(symmetric[active])),
                    "future_only_benefit_bps": float(np.nanmean(forward[active])),
                })
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "all_horizons.csv", index=False)
    combined = results[results.period == "combined_2025_2026"].copy()
    aggregate = combined.groupby("candidate", as_index=False).agg(
        horizon_case_lift_min=("case_lift", "min"),
        horizon_case_lift_mean=("case_lift", "mean"),
        horizon_corridor_lift_min=("corridor_lift_min", "min"),
        horizon_symmetric_benefit_min=("symmetric_benefit_bps", "min"),
        horizon_future_benefit_min=("future_only_benefit_bps", "min"),
        frequency_min=("frequency", "min"),
        frequency_max=("frequency", "max"),
    )
    aggregate["all_horizons_lift_1p30"] = aggregate.horizon_case_lift_min >= 1.30
    aggregate["all_horizons_symmetric_benefit_positive"] = (
        aggregate.horizon_symmetric_benefit_min > 0
    )
    aggregate["all_horizons_future_benefit_positive"] = (
        aggregate.horizon_future_benefit_min > 0
    )
    aggregate.to_csv(OUT / "candidate_summary.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CG", "horizons": HORIZONS, "fixed_policy": POLICY,
        "candidates": SOURCES,
        "model_refit": False,
        "case_lift_definition": "hit rate / expected random-day hit rate matched by corridor and calendar year",
        "case_benefit_definition": "signal day versus mean over symmetric +/-h window",
        "supplementary_benefit": "signal day versus future h publications only",
        "source_documents": [
            "Q&A для команд — сводка 20260904.pdf",
            "Q&A для команд — сводка 20260905.pdf",
        ],
        "next_cbr_rate_used": False,
        "later_period_status": "retrospective case-definition audit",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("CANDIDATE SUMMARY\n")
    print(aggregate.sort_values("horizon_case_lift_min", ascending=False).to_string(index=False))
    print("\nCOMBINED BY HORIZON\n")
    print(combined[[
        "candidate", "horizon", "frequency", "hit_rate", "case_lift",
        "pooled_lift", "corridor_lift_min", "symmetric_benefit_bps",
        "future_only_benefit_bps",
    ]].sort_values(["candidate", "horizon"]).to_string(index=False))


if __name__ == "__main__":
    main()
