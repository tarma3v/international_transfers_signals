"""Packet-EC: paired multi-horizon audit of the 15:30 fixing proxy."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round2_statistical_audit import B, SEED
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multihorizon_case_audit import corridor_period_adjusted_lift
from research.round6_multihorizon_uncertainty import _bootstrap, _holm, _weekly_stats
from research.round6_resolved_models import _fire
from research.round6_uzbek_central_bank_models import _forward


OUT = Path("results/research/round6/fixing_paired_audit")
SOURCE = Path("results/research/round6/fixing_proxies/outputs.pkl")
STANDARD = Path("results/research/round6/fixing_proxies/standard_h5_results.csv")
CANDIDATES = ("noon_consensus", "selected", "matched_stale20")
COMPARISONS = (
    ("fixing_vs_noon", "noon_consensus", "selected"),
    ("fresh_vs_stale", "matched_stale20", "selected"),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    with SOURCE.open("rb") as handle:
        outputs = pickle.load(handle)

    draws, points = {}, {}
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        symmetric = targets[f"benefit_h{h}"]
        forward = forwards[h]
        draws[h], points[h] = {}, {}
        common_valid = None
        for candidate in CANDIDATES:
            valid, fired = _fire(
                outputs[candidate], (2025, 2026), POLICY,
                y, dates, currencies,
            )
            if common_valid is None:
                common_valid = valid
            elif not np.array_equal(common_valid, valid):
                raise AssertionError(f"unaligned valid rows at h={h}")
            stats = _weekly_stats(
                y, symmetric, forward, valid, fired, dates, currencies,
            )
            lift_draw, sym_draw, fwd_draw = _bootstrap(stats, SEED)
            draws[h][candidate] = {
                "lift": lift_draw,
                "symmetric": sym_draw,
                "future": fwd_draw,
            }
            active = valid & fired
            lift, _base, _macro = corridor_period_adjusted_lift(
                y, valid, fired, currencies, dates, (2025, 2026),
            )
            points[h][candidate] = {
                "lift": lift,
                "symmetric": float(np.nanmean(symmetric[active])),
                "future": float(np.nanmean(forward[active])),
                "n": int(active.sum()),
            }

    rows = []
    for hypothesis, control, challenger in COMPARISONS:
        for h in HORIZONS:
            item = {
                "hypothesis": hypothesis,
                "horizon": h,
                "control": control,
                "challenger": challenger,
            }
            for metric in ("lift", "symmetric", "future"):
                delta = draws[h][challenger][metric] - draws[h][control][metric]
                delta = delta[np.isfinite(delta)]
                item[f"control_{metric}"] = points[h][control][metric]
                item[f"challenger_{metric}"] = points[h][challenger][metric]
                item[f"{metric}_difference"] = (
                    points[h][challenger][metric]
                    - points[h][control][metric]
                )
                item[f"{metric}_difference_ci_low"] = float(
                    np.quantile(delta, .025)
                )
                item[f"{metric}_difference_ci_high"] = float(
                    np.quantile(delta, .975)
                )
                item[f"p_{metric}_not_better"] = float(
                    (np.sum(delta <= 0.0) + 1) / (len(delta) + 1)
                )
            item["n_challenger"] = points[h][challenger]["n"]
            rows.append(item)
    audit = pd.DataFrame(rows)
    for metric in ("lift", "symmetric", "future"):
        audit[f"p_{metric}_holm_10"] = _holm(
            audit[f"p_{metric}_not_better"].to_numpy()
        )
        audit[f"{metric}_increment_supported"] = (
            audit[f"{metric}_difference_ci_low"].gt(0)
            & audit[f"p_{metric}_holm_10"].lt(.05)
        )
    audit.to_csv(OUT / "paired_horizon_audit.csv", index=False)

    lift_draws = {
        candidate: np.column_stack([
            draws[h][candidate]["lift"] for h in HORIZONS
        ])
        for candidate in CANDIDATES
    }
    aggregate_rows = []
    for hypothesis, control, challenger in COMPARISONS:
        for aggregate in ("minimum", "mean"):
            function = np.nanmin if aggregate == "minimum" else np.nanmean
            control_draw = function(lift_draws[control], axis=1)
            challenger_draw = function(lift_draws[challenger], axis=1)
            delta = challenger_draw - control_draw
            point_control = function(np.asarray([
                points[h][control]["lift"] for h in HORIZONS
            ]))
            point_challenger = function(np.asarray([
                points[h][challenger]["lift"] for h in HORIZONS
            ]))
            aggregate_rows.append({
                "hypothesis": hypothesis,
                "aggregate": aggregate,
                "control": control,
                "challenger": challenger,
                "control_lift": float(point_control),
                "challenger_lift": float(point_challenger),
                "lift_difference": float(point_challenger - point_control),
                "lift_difference_ci_low": float(np.quantile(delta, .025)),
                "lift_difference_ci_high": float(np.quantile(delta, .975)),
                "p_challenger_not_better": float(
                    (np.sum(delta <= 0.0) + 1) / (len(delta) + 1)
                ),
            })
    aggregates = pd.DataFrame(aggregate_rows)
    aggregates.to_csv(OUT / "paired_minimum_mean_audit.csv", index=False)

    standard = pd.read_csv(STANDARD)
    selected_years = standard[
        (standard.candidate == "selected")
        & standard.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    selected_combined = standard[
        (standard.candidate == "selected")
        & (standard.period == "combined_2025_2026")
    ].iloc[0]
    minimum = aggregates[
        (aggregates.hypothesis == "fixing_vs_noon")
        & (aggregates["aggregate"] == "minimum")
    ].iloc[0]
    mean = aggregates[
        (aggregates.hypothesis == "fixing_vs_noon")
        & (aggregates["aggregate"] == "mean")
    ].iloc[0]
    selected_points = audit[audit.hypothesis == "fixing_vs_noon"]
    gates = {
        "point_minimum_lift_above_noon": bool(
            minimum.challenger_lift > minimum.control_lift
        ),
        "paired_minimum_lift_lower_above_zero": bool(
            minimum.lift_difference_ci_low > 0.0
        ),
        "paired_mean_lift_lower_above_zero": bool(
            mean.lift_difference_ci_low > 0.0
        ),
        "annual_h5_lift_at_least_1p30": bool(
            selected_years.lift.ge(1.30).all()
        ),
        "annual_rate_between_1_and_2": bool(
            selected_years.frequency.between(1, 2).all()
        ),
        "minimum_currency_h5_lift_at_least_1p30": bool(
            selected_combined.corridor_lift_min >= 1.30
        ),
        "minimum_quarter_rate_at_least_1": bool(
            selected_combined.quarter_frequency_min >= 1.00
        ),
        "all_five_symmetric_benefits_positive": bool(
            selected_points.challenger_symmetric.gt(0).all()
        ),
        "all_five_future_benefits_positive": bool(
            selected_points.challenger_future.gt(0).all()
        ),
    }
    gates["promoted_over_noon"] = bool(all(gates.values()))
    fresh = aggregates[
        aggregates.hypothesis == "fresh_vs_stale"
    ].set_index("aggregate")
    freshness = {
        "minimum_lift_gain": float(fresh.loc["minimum", "lift_difference"]),
        "minimum_lift_ci_low": float(
            fresh.loc["minimum", "lift_difference_ci_low"]
        ),
        "mean_lift_gain": float(fresh.loc["mean", "lift_difference"]),
        "mean_lift_ci_low": float(fresh.loc["mean", "lift_difference_ci_low"]),
    }
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EC", "fixed_policy": POLICY,
        "candidate": "packet-EB selected geometric_mean_close",
        "comparisons": COMPARISONS,
        "horizons": HORIZONS,
        "bootstrap_draws": B,
        "bootstrap": (
            "paired moving four-week blocks, same seed/draws across horizons"
        ),
        "multiplicity": (
            "Holm separately for lift, symmetric and future differences "
            "across 10 horizon comparisons"
        ),
        "promotion_gates": gates,
        "freshness_summary": freshness,
        "model_refit": False,
        "next_cbr_rate_used": False,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(audit.to_string(index=False))
    print("\nMINIMUM/MEAN\n" + aggregates.to_string(index=False))
    print("\nGATES\n" + json.dumps(gates, indent=2))
    print("\nFRESHNESS\n" + json.dumps(freshness, indent=2))


if __name__ == "__main__":
    main()
