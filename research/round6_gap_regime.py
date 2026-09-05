"""Packet-CX: an interpretable publication-gap modifier for the incumbent."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_lifecycle import YEARS as LIFECYCLE_YEARS
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose, _forward, horizon_rows, summarize,
)


OUT = Path("results/research/round6/gap_regime")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
LIFECYCLE_PATH = Path("results/research/round6/cny_lifecycle/outputs.pkl")
LIFECYCLE = "resolved2000_handoff"
WEIGHTS = (.025, .05, .075, .10, .15, .20, .25, .30)


def _load(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def feature_output(values, template):
    """Attach a target-free row feature to exactly the template's chronology."""
    return {
        year: {
            "calib_idx": np.asarray(part["calib_idx"], dtype=int),
            "test_idx": np.asarray(part["test_idx"], dtype=int),
            "calib_score": values[np.asarray(part["calib_idx"], dtype=int)],
            "test_score": values[np.asarray(part["test_idx"], dtype=int)],
        }
        for year, part in template.items()
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    gap = X[:, names.index("gap_days")].astype(float)
    transforms = {
        "after_gap": (gap >= 3).astype(float),
        "after_long_gap": (gap >= 4).astype(float),
        "gap_days_capped": np.minimum(gap, 5.0),
        "after_gap_monday": ((gap >= 3) & np.asarray([
            day.weekday() == 0 for day in dates
        ])).astype(float),
    }
    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    candidates = {"incumbent": incumbent}
    lookup = {}
    for transform, values in transforms.items():
        part = feature_output(values, incumbent)
        for weight in WEIGHTS:
            name = f"incumbent{int(round((1-weight)*1000)):03d}_{transform}_w{int(round(weight*1000)):03d}"
            candidates[name] = combine_causal(
                [incumbent, part], (1.0 - weight, weight), dates, currencies,
            )
            lookup[name] = (transform, weight)
    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)

    comparison = {"incumbent": incumbent, "selected": candidates[selected]}
    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(comparison, years, targets, forwards, dates, currencies)
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)

    transform, weight = lookup[selected]
    lifecycle = _load(LIFECYCLE_PATH, LIFECYCLE)
    gap_lifecycle = feature_output(transforms[transform], lifecycle)
    lifecycle_gap = combine_causal(
        [lifecycle, gap_lifecycle], (1.0 - weight, weight), dates, currencies,
    )
    lifecycle_outputs = {"control": lifecycle, "gap_modifier": lifecycle_gap}
    lifecycle_horizons = horizon_rows(
        lifecycle_outputs, LIFECYCLE_YEARS, targets, forwards, dates, currencies,
    )
    lifecycle_horizons.to_csv(OUT / "lifecycle_all_horizons.csv", index=False)
    lifecycle_summary = summarize(lifecycle_horizons)
    lifecycle_summary.to_csv(OUT / "lifecycle_summary.csv", index=False)
    annual = []
    for candidate, output in lifecycle_outputs.items():
        for year in LIFECYCLE_YEARS:
            item = _evaluate(
                output, (year,), POLICY, targets["fav_h5"], forwards[5],
                dates, currencies,
            )
            item.update({"candidate": candidate, "year": year})
            annual.append(item)
    annual_frame = pd.DataFrame(annual)
    annual_frame.to_csv(OUT / "lifecycle_annual_h5.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump({**comparison, "lifecycle_gap_modifier": lifecycle_gap}, handle)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CX", "status": "post-hoc regime hypothesis",
        "selection_period": 2024, "selected": selected,
        "selected_transform": transform, "selected_weight": weight,
        "candidate_transforms": list(transforms), "weights": WEIGHTS,
        "interpretation": "a current CBR publication after a multi-day gap is known at decision time",
        "target_used_by_modifier": False,
        "full_lifecycle_years": LIFECYCLE_YEARS,
        "next_cbr_rate_used": False,
        "later_period_status": "2025-2026 already inspected before this hypothesis; prospective challenger only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected:", selected, lookup[selected])
    print("\nSCREEN TOP\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).head(20).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLIFECYCLE\n" + lifecycle_summary.to_string(index=False))
    print("\nANNUAL H5\n" + annual_frame[[
        "candidate", "year", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
