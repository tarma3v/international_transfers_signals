"""Packet-AS paired uncertainty and signal-overlap audit for local shrinkage."""
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
from research.round6_resolved_models import _fire


OUT = Path("results/research/round6/cny_local_paired_audit")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
LOCAL = Path("results/research/round6/cny_local_experts/outputs.pkl")


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
    with PRIMARY.open("rb") as handle:
        primary_output = pickle.load(handle)["logit50_extra50"]
    with LOCAL.open("rb") as handle:
        challenger_output = pickle.load(handle)["primary75_local_consensus25"]
    valid_p, primary = _fire(
        primary_output, (2025, 2026), POLICY, y, dates, currencies,
    )
    valid_c, challenger = _fire(
        challenger_output, (2025, 2026), POLICY, y, dates, currencies,
    )
    if not np.array_equal(valid_p, valid_c):
        raise AssertionError("primary and challenger rows differ")
    valid = valid_p
    draws = _bootstrap_all(
        y, benefit, dates, valid,
        {"primary": primary, "challenger": challenger},
    )
    lift_difference = draws["challenger"]["lift"] - draws["primary"]["lift"]
    benefit_difference = draws["challenger"]["benefit"] - draws["primary"]["benefit"]
    finite_lift = lift_difference[np.isfinite(lift_difference)]
    finite_benefit = benefit_difference[np.isfinite(benefit_difference)]
    summary = pd.DataFrame([{
        "primary_lift": float(y[valid & primary].mean() / y[valid].mean()),
        "challenger_lift": float(y[valid & challenger].mean() / y[valid].mean()),
        "lift_difference": float(
            y[valid & challenger].mean() / y[valid].mean()
            - y[valid & primary].mean() / y[valid].mean()
        ),
        "lift_difference_ci_low": float(np.quantile(finite_lift, .025)),
        "lift_difference_ci_high": float(np.quantile(finite_lift, .975)),
        "p_challenger_not_better": float(
            (np.sum(finite_lift <= 0) + 1) / (len(finite_lift) + 1)
        ),
        "benefit_difference_bps": float(
            np.nanmean(benefit[valid & challenger]) - np.nanmean(benefit[valid & primary])
        ),
        "benefit_difference_ci_low": float(np.quantile(finite_benefit, .025)),
        "benefit_difference_ci_high": float(np.quantile(finite_benefit, .975)),
    }])
    summary.to_csv(OUT / "paired_bootstrap.csv", index=False)
    subsets = {
        "primary": primary, "challenger": challenger,
        "intersection": primary & challenger, "union": primary | challenger,
        "primary_only": primary & ~challenger,
        "challenger_only": challenger & ~primary,
        "neither": ~primary & ~challenger,
    }
    rows = []
    for name, mask in subsets.items():
        active = valid & mask
        values = benefit[active & np.isfinite(benefit)]
        rows.append({
            "subset": name, "n": int(active.sum()),
            "share": float(active.sum() / valid.sum()),
            "target_rate": float(y[active].mean()) if active.any() else np.nan,
            "lift": float(y[active].mean() / y[valid].mean()) if active.any() else np.nan,
            "benefit_bps": float(values.mean()) if len(values) else np.nan,
        })
    overlap = pd.DataFrame(rows)
    overlap.to_csv(OUT / "signal_overlap.csv", index=False)
    jaccard = float(
        np.sum(valid & primary & challenger) / np.sum(valid & (primary | challenger))
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AS", "fixed_policy": POLICY,
        "primary": "logit50_extra50",
        "challenger": "primary75_local_consensus25",
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "jaccard": jaccard, "model_refit": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\n", overlap.to_string(index=False))
    print(f"\nJaccard={jaccard:.4f}")


if __name__ == "__main__":
    main()
