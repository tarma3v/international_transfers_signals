"""Packet-AY paired and multiplicity-aware audit of shock-bridge weights."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import B, SEED, _bootstrap_all
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_lifecycle import YEARS
from research.round6_resolved_models import _fire


OUT = Path("results/research/round6/cny_shock_weight_audit")
SOURCE = Path("results/research/round6/cny_shock_weight_plateau")
CONTROL = "cny100_anchor000"
PERIODS = {
    "shock_2022_2023": ("shock_outputs.pkl", (2022, 2023)),
    "lifecycle_2017_2026": ("lifecycle_outputs.pkl", YEARS),
}


def _load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def circular_max_difference(y, dates, currencies, valid, masks, control):
    days = np.asarray(sorted(set(dates[valid])), dtype=object)
    lookup = {(dates[row], currencies[row]): row for row in np.flatnonzero(valid)}
    target = np.full((len(days), len(CORRIDORS)), np.nan)
    fires = {name: np.zeros_like(target, dtype=bool) for name in masks}
    for i, day in enumerate(days):
        for j, currency in enumerate(CORRIDORS):
            row = lookup.get((day, currency))
            if row is None:
                continue
            target[i, j] = y[row]
            for name, mask in masks.items():
                fires[name][i, j] = mask[row]
    complete = np.all(np.isfinite(target), axis=1)
    target = target[complete]
    fires = {name: mask[complete] for name, mask in fires.items()}
    base = float(np.mean(target))
    control_lift = float(np.mean(target[fires[control]]) / base)
    observed = {
        name: float(np.mean(target[mask]) / base - control_lift)
        for name, mask in fires.items() if name != control
    }
    allowed = np.arange(20, len(target) - 20)
    rng = np.random.default_rng(SEED)
    shifts = rng.choice(allowed, size=B, replace=True)
    null = {name: np.empty(B) for name in observed}
    max_null = np.empty(B)
    for b, shift in enumerate(shifts):
        shifted = np.roll(target, int(shift), axis=0)
        shifted_base = float(np.mean(shifted))
        shifted_control = float(np.mean(shifted[fires[control]]) / shifted_base)
        values = []
        for name in observed:
            difference = float(
                np.mean(shifted[fires[name]]) / shifted_base - shifted_control
            )
            null[name][b] = difference
            values.append(difference)
        max_null[b] = max(values)
    rows = []
    for name, difference in observed.items():
        rows.append({
            "candidate": name,
            "observed_lift_difference": difference,
            "circular_p_unadjusted": float(
                (np.sum(null[name] >= difference) + 1) / (B + 1)
            ),
            "circular_p_max_adjusted": float(
                (np.sum(max_null >= difference) + 1) / (B + 1)
            ),
            "null_max_difference_q95": float(np.quantile(max_null, .95)),
            "n_challengers": len(observed),
        })
    return pd.DataFrame(rows)


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
    summary_frames = []
    for period, (filename, years) in PERIODS.items():
        outputs = _load(SOURCE / filename)
        masks, common_valid = {}, None
        for name, output in outputs.items():
            valid, fired = _fire(output, years, POLICY, y, dates, currencies)
            if common_valid is None:
                common_valid = valid
            elif not np.array_equal(common_valid, valid):
                raise AssertionError(f"comparison rows differ in {period}")
            masks[name] = fired
        draws = _bootstrap_all(y, benefit, dates, common_valid, masks)
        control_lift = float(y[common_valid & masks[CONTROL]].mean() / y[common_valid].mean())
        control_benefit = float(np.nanmean(benefit[common_valid & masks[CONTROL]]))
        rows = []
        for name, fired in masks.items():
            if name == CONTROL:
                continue
            lift = float(y[common_valid & fired].mean() / y[common_valid].mean())
            lift_difference = draws[name]["lift"] - draws[CONTROL]["lift"]
            benefit_difference = draws[name]["benefit"] - draws[CONTROL]["benefit"]
            finite_lift = lift_difference[np.isfinite(lift_difference)]
            finite_benefit = benefit_difference[np.isfinite(benefit_difference)]
            rows.append({
                "period": period,
                "candidate": name,
                "control_lift": control_lift,
                "challenger_lift": lift,
                "lift_difference": lift - control_lift,
                "lift_difference_ci_low": float(np.quantile(finite_lift, .025)),
                "lift_difference_ci_high": float(np.quantile(finite_lift, .975)),
                "p_challenger_not_better": float(
                    (np.sum(finite_lift <= 0) + 1) / (len(finite_lift) + 1)
                ),
                "control_benefit_bps": control_benefit,
                "challenger_benefit_bps": float(np.nanmean(benefit[common_valid & fired])),
                "benefit_difference_ci_low": float(np.quantile(finite_benefit, .025)),
                "benefit_difference_ci_high": float(np.quantile(finite_benefit, .975)),
            })
        paired = pd.DataFrame(rows)
        circular = circular_max_difference(
            y, dates, currencies, common_valid, masks, CONTROL,
        )
        paired = paired.merge(circular, on="candidate", how="left")
        summary_frames.append(paired)
    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(OUT / "paired_multiplicity_audit.csv", index=False)
    candidate = "cny060_anchor040"
    selected = summary[summary.candidate == candidate]
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "AY",
        "control": CONTROL,
        "grid_identified_candidate": candidate,
        "fixed_policy": POLICY,
        "bootstrap_draws": B,
        "bootstrap": "paired four-week blocks with currencies kept by date",
        "multiplicity": "max lift difference over six candidates under circular date shifts",
        "ordinary_shock_interval_excludes_zero": bool(
            selected[selected.period == "shock_2022_2023"]
            .lift_difference_ci_low.iloc[0] > 0
        ),
        "shock_max_adjusted_p_below_005": bool(
            selected[selected.period == "shock_2022_2023"]
            .circular_p_max_adjusted.iloc[0] < .05
        ),
        "model_refit": False,
        "later_period_status": "multiplicity-aware retrospective audit",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.sort_values(["period", "lift_difference"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
