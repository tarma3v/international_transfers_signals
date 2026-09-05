"""Packet-AW paired audit of the low-dose anchor shock bridge."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _bootstrap_all
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_lifecycle import YEARS
from research.round6_resolved_models import _fire


OUT = Path("results/research/round6/cny_lifecycle_paired_audit")
SOURCE = Path("results/research/round6/cny_shock_blends")
CONTROL = "cny_expanding"
CHALLENGER = "cny75_anchor25"


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
    sources = {
        "shock_2022_2023": (
            _load(SOURCE / "shock_outputs.pkl"), (2022, 2023),
        ),
        "lifecycle_2017_2026": (
            _load(SOURCE / "lifecycle_outputs.pkl"), YEARS,
        ),
    }
    summaries, overlaps, jaccards = [], [], {}
    for period, (outputs, years) in sources.items():
        valid_p, control = _fire(
            outputs[CONTROL], years, POLICY, y, dates, currencies,
        )
        valid_c, challenger = _fire(
            outputs[CHALLENGER], years, POLICY, y, dates, currencies,
        )
        if not np.array_equal(valid_p, valid_c):
            raise AssertionError(f"comparison rows differ in {period}")
        valid = valid_p
        draws = _bootstrap_all(
            y, benefit, dates, valid,
            {"control": control, "challenger": challenger},
        )
        lift_difference = draws["challenger"]["lift"] - draws["control"]["lift"]
        benefit_difference = (
            draws["challenger"]["benefit"] - draws["control"]["benefit"]
        )
        finite_lift = lift_difference[np.isfinite(lift_difference)]
        finite_benefit = benefit_difference[np.isfinite(benefit_difference)]
        control_lift = float(y[valid & control].mean() / y[valid].mean())
        challenger_lift = float(y[valid & challenger].mean() / y[valid].mean())
        summaries.append({
            "period": period,
            "control_lift": control_lift,
            "challenger_lift": challenger_lift,
            "lift_difference": challenger_lift - control_lift,
            "lift_difference_ci_low": float(np.quantile(finite_lift, .025)),
            "lift_difference_ci_high": float(np.quantile(finite_lift, .975)),
            "p_challenger_not_better": float(
                (np.sum(finite_lift <= 0) + 1) / (len(finite_lift) + 1)
            ),
            "benefit_difference_bps": float(
                np.nanmean(benefit[valid & challenger])
                - np.nanmean(benefit[valid & control])
            ),
            "benefit_difference_ci_low": float(np.quantile(finite_benefit, .025)),
            "benefit_difference_ci_high": float(np.quantile(finite_benefit, .975)),
        })
        subsets = {
            "control": control,
            "challenger": challenger,
            "intersection": control & challenger,
            "union": control | challenger,
            "control_only": control & ~challenger,
            "challenger_only": challenger & ~control,
            "neither": ~control & ~challenger,
        }
        for name, mask in subsets.items():
            active = valid & mask
            values = benefit[active & np.isfinite(benefit)]
            overlaps.append({
                "period": period,
                "subset": name,
                "n": int(active.sum()),
                "share": float(active.sum() / valid.sum()),
                "target_rate": float(y[active].mean()) if active.any() else np.nan,
                "lift": (
                    float(y[active].mean() / y[valid].mean())
                    if active.any() else np.nan
                ),
                "benefit_bps": float(values.mean()) if len(values) else np.nan,
            })
        jaccards[period] = float(
            np.sum(valid & control & challenger)
            / np.sum(valid & (control | challenger))
        )
    summary = pd.DataFrame(summaries)
    overlap = pd.DataFrame(overlaps)
    summary.to_csv(OUT / "paired_bootstrap.csv", index=False)
    overlap.to_csv(OUT / "signal_overlap.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AW",
        "fixed_policy": POLICY,
        "control": CONTROL,
        "challenger": CHALLENGER,
        "periods": {"shock_2022_2023": [2022, 2023],
                    "lifecycle_2017_2026": list(YEARS)},
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "jaccard": jaccards,
        "model_refit": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\n" + overlap.to_string(index=False))


if __name__ == "__main__":
    main()
