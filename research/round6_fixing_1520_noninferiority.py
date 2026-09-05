"""Packet-EK: paired non-inferiority audit for the 15:20 fixing router."""
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
from research.round6_multihorizon_uncertainty import _bootstrap, _weekly_stats
from research.round6_resolved_models import _fire
from research.round6_uzbek_central_bank_models import _forward


OUT = Path("results/research/round6/fixing_1520_noninferiority")
SOURCE = Path("results/research/round6/fixing_1520_router/outputs.pkl")
STANDARD = Path("results/research/round6/fixing_1520_router/standard_h5_results.csv")
CANDIDATES = ("route_1530", "route_1520", "matched_stale20")
COMPARISONS = (
    ("early_vs_late", "route_1530", "route_1520"),
    ("fresh_vs_stale", "matched_stale20", "route_1520"),
)
LIFT_MARGIN = .05
BENEFIT_MARGIN_BPS = 5.0


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
                point_delta = (
                    points[h][challenger][metric] - points[h][control][metric]
                )
                margin = LIFT_MARGIN if metric == "lift" else BENEFIT_MARGIN_BPS
                item[f"control_{metric}"] = points[h][control][metric]
                item[f"challenger_{metric}"] = points[h][challenger][metric]
                item[f"{metric}_difference"] = point_delta
                item[f"{metric}_difference_ci_low"] = float(
                    np.quantile(delta, .025)
                )
                item[f"{metric}_difference_ci_high"] = float(
                    np.quantile(delta, .975)
                )
                item[f"p_{metric}_below_negative_margin"] = float(
                    (np.sum(delta <= -margin) + 1) / (len(delta) + 1)
                )
                item[f"{metric}_noninferior"] = bool(
                    item[f"{metric}_difference_ci_low"] > -margin
                )
            item["n_challenger"] = points[h][challenger]["n"]
            rows.append(item)
    audit = pd.DataFrame(rows)
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
            low = float(np.quantile(delta, .025))
            aggregate_rows.append({
                "hypothesis": hypothesis,
                "aggregate": aggregate,
                "control": control,
                "challenger": challenger,
                "control_lift": float(point_control),
                "challenger_lift": float(point_challenger),
                "lift_difference": float(point_challenger - point_control),
                "lift_difference_ci_low": low,
                "lift_difference_ci_high": float(np.quantile(delta, .975)),
                "p_below_negative_margin": float(
                    (np.sum(delta <= -LIFT_MARGIN) + 1) / (len(delta) + 1)
                ),
                "noninferior": bool(low > -LIFT_MARGIN),
            })
    aggregates = pd.DataFrame(aggregate_rows)
    aggregates.to_csv(OUT / "paired_minimum_mean_audit.csv", index=False)

    standard = pd.read_csv(STANDARD)
    years = standard[
        (standard.candidate == "route_1520")
        & standard.period.isin(("retrospective_2025", "retrospective_2026"))
    ]
    combined = standard[
        (standard.candidate == "route_1520")
        & (standard.period == "combined_2025_2026")
    ].iloc[0]
    early = audit[audit.hypothesis == "early_vs_late"]
    early_aggregates = aggregates[aggregates.hypothesis == "early_vs_late"]
    gates = {
        "both_aggregate_lifts_noninferior": bool(
            early_aggregates.noninferior.all()
        ),
        "every_horizon_lift_noninferior": bool(early.lift_noninferior.all()),
        "every_horizon_symmetric_benefit_noninferior": bool(
            early.symmetric_noninferior.all()
        ),
        "every_horizon_future_benefit_noninferior": bool(
            early.future_noninferior.all()
        ),
        "annual_h5_rate_between_1_and_2": bool(
            years.frequency.between(1.0, 2.0).all()
        ),
        "minimum_currency_h5_lift_at_least_1p30": bool(
            combined.corridor_lift_min >= 1.30
        ),
        "minimum_quarter_rate_at_least_0p95": bool(
            combined.quarter_frequency_min >= .95
        ),
    }
    gates["earlier_product_noninferior"] = bool(all(gates.values()))
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EK",
        "candidate": "packet-EJ route_1520",
        "control": "packet-ED route_1530",
        "comparisons": COMPARISONS,
        "horizons": HORIZONS,
        "lift_noninferiority_margin": LIFT_MARGIN,
        "benefit_noninferiority_margin_bps": BENEFIT_MARGIN_BPS,
        "bootstrap_draws": B,
        "bootstrap": (
            "paired moving four-week blocks, same seed/draws across horizons"
        ),
        "noninferiority_gates": gates,
        "model_refit": False,
        "next_cbr_rate_used": False,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nHORIZONS\n" + audit.to_string(index=False))
    print("\nMINIMUM/MEAN\n" + aggregates.to_string(index=False))
    print("\nGATES\n" + json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
