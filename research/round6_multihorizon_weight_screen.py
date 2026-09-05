"""Packet-CI: choose a transparent three-way blend on 2024 worst-horizon lift."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, benefit_forward_only, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multihorizon_case_audit import corridor_period_adjusted_lift
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate, _fire


OUT = Path("results/research/round6/multihorizon_weight_screen")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
GEOMETRY = Path("results/research/round6/cny_expert_geometry/outputs.pkl")
CBA = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"


def _load(path, name):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def weight_grid():
    result = []
    for p in range(4, 9):
        for g in range(1, 10 - p):
            c = 10 - p - g
            if c >= 1:
                result.append((p / 10.0, g / 10.0, c / 10.0))
    return result


def _forward(series, index, h):
    result = np.full(len(index), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, position, h)
        if value is not None:
            result[row] = value
    return result


def horizon_rows(candidate, output, years, period, targets, series, index, dates, currencies):
    rows = []
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        symmetric = targets[f"benefit_h{h}"]
        forward = _forward(series, index, h)
        valid, fired = _fire(output, years, POLICY, y, dates, currencies)
        active = valid & fired
        case_lift, matched_base, macro = corridor_period_adjusted_lift(
            y, valid, fired, currencies, dates, years,
        )
        rows.append({
            "candidate": candidate, "period": period, "horizon": h,
            "n": int(active.sum()), "case_lift": case_lift,
            "matched_base_rate": matched_base,
            "macro_corridor_year_lift": macro,
            "symmetric_benefit_bps": float(np.nanmean(symmetric[active])),
            "future_only_benefit_bps": float(np.nanmean(forward[active])),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    benefit5 = _forward(series, index, 5)
    primary = _load(PRIMARY, "logit50_extra50")
    geometry = _load(GEOMETRY, "geometry_min75_max25")
    cba = _load(CBA, "cba_consensus_basis")
    outputs = {}
    weights_by_name = {}
    for weights in weight_grid():
        name = "threeway_p{:02d}_g{:02d}_a{:02d}".format(
            *(int(round(value * 100)) for value in weights)
        )
        outputs[name] = combine_causal(
            [primary, geometry, cba], weights, dates, currencies,
        )
        weights_by_name[name] = weights
    screen_detail = []
    for candidate, output in outputs.items():
        screen_detail.extend(horizon_rows(
            candidate, output, (2024,), "screen_2024", targets,
            series, index, dates, currencies,
        ))
    detail = pd.DataFrame(screen_detail)
    screen = detail.groupby("candidate", as_index=False).agg(
        horizon_lift_min=("case_lift", "min"),
        horizon_lift_mean=("case_lift", "mean"),
        symmetric_benefit_min=("symmetric_benefit_bps", "min"),
        future_benefit_min=("future_only_benefit_bps", "min"),
    )
    feasible = screen[
        screen.symmetric_benefit_min.gt(0)
        & screen.future_benefit_min.gt(0)
    ].copy()
    pool = feasible if len(feasible) else screen.copy()
    selected = str(pool.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "symmetric_benefit_min"],
        ascending=False,
    ).iloc[0].candidate)
    detail.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen.to_csv(OUT / "screen_2024_summary.csv", index=False)

    incumbent = _load(CBA, INCUMBENT)
    comparison_outputs = {"selected_threeway": outputs[selected], "incumbent_nested": incumbent}
    later_rows = []
    for candidate, output in comparison_outputs.items():
        for period, years in (
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            later_rows.extend(horizon_rows(
                candidate, output, years, period, targets,
                series, index, dates, currencies,
            ))
    later = pd.DataFrame(later_rows)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)

    # Standard h=5 audit is retained for comparability with prior packets.
    standard = []
    for candidate, output in comparison_outputs.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(output, years, POLICY, y5, benefit5, dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            standard.append(item)
    standard = pd.DataFrame(standard)
    standard.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        standard[standard.period == "screen_2024"], comparison_outputs,
        (2025, 2026), y5, benefit5, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "multihorizon_weight_screen_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison_outputs.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, benefit5, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    later_summary = later[later.period == "combined_2025_2026"].groupby(
        "candidate", as_index=False,
    ).agg(
        horizon_lift_min=("case_lift", "min"),
        horizon_lift_mean=("case_lift", "mean"),
        symmetric_benefit_min=("symmetric_benefit_bps", "min"),
        future_benefit_min=("future_only_benefit_bps", "min"),
    )
    later_summary.to_csv(OUT / "later_summary.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CI", "fixed_policy": POLICY, "selection_period": 2024,
        "selection_objective": "maximum worst case-lift over h=1/3/5/10/20",
        "selection_constraints": "positive symmetric and future-only benefit at every h",
        "weight_grid": weight_grid(), "selected": selected,
        "selected_weights_primary_geometry_armenian": weights_by_name[selected],
        "components": [
            "logit50_extra50", "geometry_min75_max25", "cba_consensus_basis",
        ],
        "model_refit": False, "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected on 2024: {selected} weights={weights_by_name[selected]}\n")
    print(screen.sort_values("horizon_lift_min", ascending=False).head(10).to_string(index=False))
    print("\nLATER SUMMARY\n" + later_summary.to_string(index=False))
    print("\nH5 STANDARD\n" + standard.to_string(index=False))


if __name__ == "__main__":
    main()
