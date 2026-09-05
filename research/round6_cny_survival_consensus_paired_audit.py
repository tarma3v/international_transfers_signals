"""Packet-BW paired audit of fixed survival-expert consensuses."""
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


OUT = Path("results/research/round6/cny_survival_consensus_paired_audit")
CONSENSUS = Path("results/research/round6/cny_survival_consensus/outputs.pkl")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
SHADOW = Path("results/research/round6/cny_shadow_nowcast/outputs.pkl")
SURVIVAL = Path("results/research/round6/cny_survival_hazard/outputs.pkl")
COMPARISONS = (
    ("primary_survival_increment", "logit50_extra50", "primary75_survival25"),
    ("shadow_survival_vs_shadow", "shadow_close_basis", "shadow75_survival25"),
    ("shadow_survival_vs_survival", "survival_cumulative_geometric", "shadow75_survival25"),
    ("three_expert_increment", "logit50_extra50", "primary50_survival25_shadow25"),
)


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _holm(values):
    order = np.argsort(values)
    out = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[position])))
        out[position] = running
    return out


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
    outputs = _load(CONSENSUS)
    outputs["logit50_extra50"] = _load(PRIMARY)["logit50_extra50"]
    outputs["shadow_close_basis"] = _load(SHADOW)["shadow_close_basis"]
    outputs["survival_cumulative_geometric"] = _load(SURVIVAL)[
        "survival_cumulative_geometric"
    ]
    rows, overlaps = [], []
    for hypothesis, control, challenger in COMPARISONS:
        valid_c, fired_c = _fire(
            outputs[control], (2025, 2026), POLICY, y, dates, currencies,
        )
        valid_h, fired_h = _fire(
            outputs[challenger], (2025, 2026), POLICY, y, dates, currencies,
        )
        if not np.array_equal(valid_c, valid_h):
            raise AssertionError(f"comparison rows differ for {hypothesis}")
        valid = valid_c
        masks = {control: fired_c, challenger: fired_h}
        draws = _bootstrap_all(y, benefit, dates, valid, masks)
        lift_delta = draws[challenger]["lift"] - draws[control]["lift"]
        benefit_delta = draws[challenger]["benefit"] - draws[control]["benefit"]
        lift_delta = lift_delta[np.isfinite(lift_delta)]
        benefit_delta = benefit_delta[np.isfinite(benefit_delta)]
        control_lift = float(y[valid & fired_c].mean() / y[valid].mean())
        challenger_lift = float(y[valid & fired_h].mean() / y[valid].mean())
        control_benefit = float(np.nanmean(benefit[valid & fired_c]))
        challenger_benefit = float(np.nanmean(benefit[valid & fired_h]))
        circular = circular_max_difference(
            y, dates, currencies, valid, masks, control,
        ).iloc[0]
        rows.append({
            "hypothesis": hypothesis,
            "control": control,
            "challenger": challenger,
            "control_lift": control_lift,
            "challenger_lift": challenger_lift,
            "lift_difference": challenger_lift - control_lift,
            "lift_difference_ci_low": float(np.quantile(lift_delta, .025)),
            "lift_difference_ci_high": float(np.quantile(lift_delta, .975)),
            "p_challenger_not_better": float(
                (np.sum(lift_delta <= 0) + 1) / (len(lift_delta) + 1)
            ),
            "control_benefit_bps": control_benefit,
            "challenger_benefit_bps": challenger_benefit,
            "benefit_difference_bps": challenger_benefit - control_benefit,
            "benefit_difference_ci_low": float(np.quantile(benefit_delta, .025)),
            "benefit_difference_ci_high": float(np.quantile(benefit_delta, .975)),
            "circular_p_unadjusted": float(circular.circular_p_unadjusted),
        })
        for subset, mask in {
            "control": fired_c,
            "challenger": fired_h,
            "intersection": fired_c & fired_h,
            "control_only": fired_c & ~fired_h,
            "challenger_only": fired_h & ~fired_c,
            "neither": ~fired_c & ~fired_h,
        }.items():
            active = valid & mask
            values = benefit[active & np.isfinite(benefit)]
            overlaps.append({
                "hypothesis": hypothesis,
                "subset": subset,
                "n": int(active.sum()),
                "share": float(active.sum() / valid.sum()),
                "lift": float(y[active].mean() / y[valid].mean()) if active.any() else np.nan,
                "benefit_bps": float(values.mean()) if len(values) else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary["p_holm_four_hypotheses"] = _holm(
        summary.p_challenger_not_better.to_numpy()
    )
    summary["lift_ci_above_zero"] = summary.lift_difference_ci_low > 0
    summary["benefit_ci_above_zero"] = summary.benefit_difference_ci_low > 0
    summary["increment_supported"] = (
        summary.lift_ci_above_zero & summary.benefit_ci_above_zero
        & summary.p_holm_four_hypotheses.lt(.05)
    )
    summary.to_csv(OUT / "paired_audit.csv", index=False)
    pd.DataFrame(overlaps).to_csv(OUT / "signal_overlap.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BW",
        "comparisons": COMPARISONS,
        "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "Holm adjustment across four predeclared hypotheses",
        "any_increment_supported": bool(summary.increment_supported.any()),
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
