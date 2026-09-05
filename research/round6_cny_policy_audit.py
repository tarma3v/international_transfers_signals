"""Packet-AJ local policy plateau and component-overlap audit."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_resolved_models import _evaluate, _fire


OUT = Path("results/research/round6/cny_policy_audit")
CONSENSUS = Path("results/research/round6/cny_consensus/outputs.pkl")
COMPONENTS = Path("results/research/round6/cny_explainable/outputs.pkl")
RATES = (.18, .20, .22, .25, .30)
WINDOWS = (20, 40, 60)


def policy(rate, rolling):
    return {**POLICY, "rate": rate, "rolling": rolling}


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
    with CONSENSUS.open("rb") as handle:
        consensus = pickle.load(handle)["logit50_extra50"]
    with COMPONENTS.open("rb") as handle:
        components = pickle.load(handle)

    rows = []
    for rate in RATES:
        for rolling in WINDOWS:
            current = policy(rate, rolling)
            for period, years in (
                ("screen_2024", (2024,)),
                ("retrospective_2025", (2025,)),
                ("retrospective_2026", (2026,)),
            ):
                item = _evaluate(consensus, years, current,
                                 y, benefit, dates, currencies)
                item.update({"period": period, **current})
                rows.append(item)
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUT / "policy_sensitivity.csv", index=False)
    piv = sensitivity.pivot_table(
        index=["rate", "rolling"], columns="period",
        values=["lift", "frequency", "corridor_lift_min", "quarter_frequency_min"],
    )
    pass_mask = (
        piv[("lift", "retrospective_2025")].ge(1.30)
        & piv[("lift", "retrospective_2026")].ge(1.30)
        & piv[("frequency", "retrospective_2025")].between(1.0, 2.0)
        & piv[("frequency", "retrospective_2026")].between(1.0, 2.0)
    )
    plateau = piv.reset_index()
    plateau["passes_both_years"] = pass_mask.to_numpy()
    plateau.to_csv(OUT / "policy_plateau.csv", index=False)

    valid, ensemble_fire = _fire(
        consensus, (2025, 2026), POLICY, y, dates, currencies,
    )
    valid_l, logit_fire = _fire(
        components["market_anchor_logit"], (2025, 2026), POLICY,
        y, dates, currencies,
    )
    valid_e, extra_fire = _fire(
        components["cny_intraday_extra"], (2025, 2026), POLICY,
        y, dates, currencies,
    )
    if not np.array_equal(valid, valid_l) or not np.array_equal(valid, valid_e):
        raise AssertionError("component evaluation rows differ")
    subsets = {
        "logit": logit_fire,
        "extra": extra_fire,
        "ensemble": ensemble_fire,
        "intersection": logit_fire & extra_fire,
        "union": logit_fire | extra_fire,
        "logit_only": logit_fire & ~extra_fire,
        "extra_only": extra_fire & ~logit_fire,
        "neither": ~logit_fire & ~extra_fire,
    }
    overlap_rows = []
    for name, fired in subsets.items():
        active = valid & fired
        b = benefit[active & np.isfinite(benefit)]
        overlap_rows.append({
            "subset": name, "n": int(active.sum()),
            "share_of_valid": float(active.sum() / valid.sum()),
            "target_rate": float(y[active].mean()) if active.any() else np.nan,
            "lift": float(y[active].mean() / y[valid].mean()) if active.any() else np.nan,
            "benefit_bps": float(b.mean()) if len(b) else np.nan,
        })
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(OUT / "component_overlap.csv", index=False)
    jaccard = float(
        np.sum(valid & logit_fire & extra_fire)
        / np.sum(valid & (logit_fire | extra_fire))
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AJ", "primary_candidate": "logit50_extra50",
        "primary_policy_unchanged": POLICY, "rates": RATES, "windows": WINDOWS,
        "grid_used_for_selection": False, "jaccard_logit_extra": jaccard,
        "plateau_cells": int(len(plateau)),
        "cells_passing_both_lift_and_rate": int(pass_mask.sum()),
        "later_period_status": "post-diagnostic retrospective sensitivity",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPOLICY PLATEAU\n" + plateau.to_string(index=False))
    print("\nOVERLAP\n" + overlap.to_string(index=False))
    print(f"\nPassing cells: {int(pass_mask.sum())}/{len(pass_mask)}, Jaccard={jaccard:.3f}")


if __name__ == "__main__":
    main()
