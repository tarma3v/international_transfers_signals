"""Causal weekly alert-budget controllers for frozen round-five scores."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _bootstrap_all, _circular_shift_audit
from research.round5_features import load_round5_features


OUT = Path("results/research/round6/rate_control")
SOURCE = Path("results/research/round5/adaptation/outputs.pkl")
SEED = 20260905


@dataclass(frozen=True)
class Policy:
    history: int
    strong: float
    late: float
    late_weekday: int
    weekly_cap: int


def policies():
    return [
        Policy(history, strong, late, late_weekday, cap)
        for history in (10, 20, 40, 60)
        for strong in (.75, .80, .85, .90)
        for late in (.00, .30, .50, .65)
        for late_weekday in (3, 4)
        for cap in (1, 2)
        if late <= strong
    ]


def _rank(reference, value):
    if len(reference) < 5:
        return .5
    ordered = np.sort(np.asarray(reference, dtype=float))
    return float(np.searchsorted(ordered, value, side="right") / len(ordered))


def controlled_fired(output, years, dates, currencies, y, policy):
    valid = np.zeros(len(y), dtype=bool)
    fired = np.zeros(len(y), dtype=bool)
    score = np.full(len(y), np.nan)
    for year in years:
        z = output[year]
        te = np.asarray(z["test_idx"], dtype=int)
        valid[te] = ~np.isnan(y[te])
        score[te] = np.asarray(z["test_score"], dtype=float)
    for currency in CORRIDORS:
        rows = np.where(valid & (currencies == currency))[0]
        rows = rows[np.argsort(dates[rows])]
        history = []
        quarter_key = None
        week_key = None
        week_count = 0
        for row in rows:
            day = dates[row]
            current_quarter = (day.year, (day.month - 1) // 3 + 1)
            if current_quarter != quarter_key:
                history = []
                quarter_key = current_quarter
            iso = day.isocalendar()
            current_week = (iso[0], iso[1])
            if current_week != week_key:
                week_key = current_week
                week_count = 0
            percentile = _rank(history[-policy.history:], score[row])
            threshold = policy.strong
            if week_count == 0 and day.weekday() >= policy.late_weekday:
                threshold = min(threshold, policy.late)
            if week_count < policy.weekly_cap and percentile >= threshold:
                fired[row] = True
                week_count += 1
            history.append(float(score[row]))
    return valid, fired, score


def _metrics(output, years, policy, y, benefit, dates, currencies):
    valid, fired, score = controlled_fired(
        output, years, dates, currencies, y, policy,
    )
    active = valid & fired
    base = float(y[valid].mean())
    hit = float(y[active].mean()) if active.any() else np.nan
    b = benefit[active & ~np.isnan(benefit)]
    year_lifts, year_freq = [], []
    for year in years:
        scope = valid & np.asarray([day.year == year for day in dates])
        signals = active & scope
        if scope.any() and signals.any():
            year_lifts.append(float(y[signals].mean() / y[scope].mean()))
            year_freq.append(rate_per_week(
                int(signals.sum()), len(CORRIDORS), dates, scope,
            ))
    corridor_lifts, corridor_freq = [], []
    for currency in CORRIDORS:
        scope = valid & (currencies == currency)
        signals = active & (currencies == currency)
        corridor_lifts.append(
            float(y[signals].mean() / y[scope].mean()) if signals.any() else np.nan
        )
        corridor_freq.append(rate_per_week(
            int(signals.sum()), 1, dates, scope,
        ))
    quarter_freq = []
    for year in years:
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                day.year == year and (day.month - 1) // 3 + 1 == quarter
                for day in dates
            ])
            if scope.any():
                quarter_freq.append(rate_per_week(
                    int((active & scope).sum()), len(CORRIDORS), dates, scope,
                ))
    clustered = []
    for currency in CORRIDORS:
        signal_dates = sorted(dates[active & (currencies == currency)])
        clustered.extend([
            (right - left).days <= 7 for left, right in zip(signal_dates[:-1], signal_dates[1:])
        ])
    return {
        "n": int(active.sum()),
        "frequency": rate_per_week(int(active.sum()), len(CORRIDORS), dates, valid),
        "hit_rate": hit, "base_rate": base, "lift": hit / base,
        "auc": float(roc_auc_score(y[valid], score[valid])),
        "forward_benefit_bps": float(np.mean(b)) if len(b) else np.nan,
        "macro_year_lift": float(np.mean(year_lifts)),
        "year_lift_min": float(min(year_lifts)),
        "year_frequency_min": float(min(year_freq)),
        "year_frequency_max": float(max(year_freq)),
        "corridor_lift_min": float(np.nanmin(corridor_lifts)),
        "corridor_freq_min": float(min(corridor_freq)),
        "corridor_freq_max": float(max(corridor_freq)),
        "quarter_frequency_min": float(min(quarter_freq)),
        "quarter_frequency_max": float(max(quarter_freq)),
        "cluster_share_7d": float(np.mean(clustered)) if clustered else np.nan,
    }


def _choose(part):
    feasible = part[
        part.frequency.between(1.00, 2.00)
        & part.corridor_freq_min.ge(.80)
        & part.quarter_frequency_min.ge(.70)
        & part.forward_benefit_bps.gt(0)
    ].copy()
    pool = feasible if len(feasible) else part.copy()
    pool["robustness"] = pool[["lift", "corridor_lift_min"]].min(axis=1)
    return pool.sort_values(
        ["robustness", "lift", "quarter_frequency_min"], ascending=False,
    ).iloc[0]


def _breakdown(candidate, output, years, policy, y, benefit, dates, currencies):
    valid, fired, _score = controlled_fired(
        output, years, dates, currencies, y, policy,
    )
    groups = [("overall", "all", valid, len(CORRIDORS))]
    for year in years:
        groups.append((
            "year", str(year),
            valid & np.asarray([day.year == year for day in dates]), len(CORRIDORS),
        ))
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                day.year == year and (day.month - 1) // 3 + 1 == quarter
                for day in dates
            ])
            if scope.any():
                groups.append(("quarter", f"{year}Q{quarter}", scope, len(CORRIDORS)))
    for currency in CORRIDORS:
        groups.append(("currency", currency, valid & (currencies == currency), 1))
    rows = []
    for kind, group, scope, n_corridors in groups:
        active = scope & fired
        values = benefit[active & ~np.isnan(benefit)]
        base = float(y[scope].mean())
        hit = float(y[active].mean()) if active.any() else np.nan
        rows.append({
            "candidate": candidate, "breakdown": kind, "group": group,
            "n_scope": int(scope.sum()), "n_signals": int(active.sum()),
            "frequency": rate_per_week(
                int(active.sum()), n_corridors, dates, scope,
            ),
            "base_rate": base, "hit_rate": hit,
            "lift": hit / base if active.any() else np.nan,
            "forward_benefit_bps": float(np.mean(values)) if len(values) else np.nan,
        })
    return rows


def _bootstrap(selected, outputs, years, y, benefit, dates, currencies):
    valid = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
    masks = {}
    for row in selected.itertuples(index=False):
        policy = Policy(
            int(row.history), float(row.strong), float(row.late),
            int(row.late_weekday), int(row.weekly_cap),
        )
        actual, fired, _score = controlled_fired(
            outputs[row.candidate], years, dates, currencies, y, policy,
        )
        if np.array_equal(actual, valid):
            masks[row.candidate] = fired
    draws = _bootstrap_all(y, benefit, dates, valid, masks)
    rows = []
    for name, fired in masks.items():
        active = valid & fired
        lift_draws = draws[name]["lift"]
        benefit_draws = draws[name]["benefit"]
        lift_draws = lift_draws[np.isfinite(lift_draws)]
        benefit_draws = benefit_draws[np.isfinite(benefit_draws)]
        rows.append({
            "candidate": name, "n": int(active.sum()),
            "lift": float(y[active].mean() / y[valid].mean()),
            "lift_ci_low": float(np.quantile(lift_draws, .025)),
            "lift_ci_high": float(np.quantile(lift_draws, .975)),
            "p_lift_le_1": float((np.sum(lift_draws <= 1.0) + 1) / (len(lift_draws) + 1)),
            "benefit_ci_low": float(np.quantile(benefit_draws, .025)),
            "benefit_ci_high": float(np.quantile(benefit_draws, .975)),
        })
    return pd.DataFrame(rows), masks, valid


def main():
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
    with SOURCE.open("rb") as handle:
        all_outputs = pickle.load(handle)
    names = (
        "quarterly_reset_hist", "anchor_multiscale_locked",
        "quarterly_reset_hist_anchor50",
    )
    outputs = {name: all_outputs[name] for name in names}

    grid_rows = []
    for name, output in outputs.items():
        for policy in policies():
            item = _metrics(output, (2024,), policy, y, benefit, dates, currencies)
            item.update({"candidate": name, **asdict(policy)})
            grid_rows.append(item)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "screen_2024_grid.csv", index=False)
    selected = pd.DataFrame([_choose(part) for _, part in grid.groupby("candidate")])
    selected = selected.sort_values(["robustness", "lift"], ascending=False)
    selected.to_csv(OUT / "screen_2024_selected.csv", index=False)

    confirm_rows, audit_rows, combined_rows = [], [], []
    for row in selected.itertuples(index=False):
        policy = Policy(
            int(row.history), float(row.strong), float(row.late),
            int(row.late_weekday), int(row.weekly_cap),
        )
        for years, target in (((2025,), confirm_rows), ((2026,), audit_rows),
                              ((2025, 2026), combined_rows)):
            item = _metrics(
                outputs[row.candidate], years, policy, y, benefit, dates, currencies,
            )
            item.update({"candidate": row.candidate, **asdict(policy)})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            if years == (2025,):
                item["clears_1p30_gate"] = bool(
                    item["lift"] >= 1.30 and 1.00 <= item["frequency"] <= 2.00
                    and item["corridor_freq_min"] >= .80
                    and item["quarter_frequency_min"] >= .70
                    and item["forward_benefit_bps"] > 0
                )
            target.append(item)
    confirm = pd.DataFrame(confirm_rows).sort_values(
        ["clears_1p30_gate", "robustness", "lift"], ascending=False,
    )
    passed = set(confirm.loc[confirm.clears_1p30_gate, "candidate"])
    audit = pd.DataFrame(audit_rows)
    audit["passed_2025"] = audit.candidate.isin(passed)
    audit = audit.sort_values(["passed_2025", "robustness", "lift"], ascending=False)
    combined = pd.DataFrame(combined_rows)
    combined["passed_2025"] = combined.candidate.isin(passed)
    combined = combined.sort_values(
        ["passed_2025", "macro_year_lift", "lift"], ascending=False,
    )
    confirm.to_csv(OUT / "confirm_2025.csv", index=False)
    audit.to_csv(OUT / "audit_2026.csv", index=False)
    combined.to_csv(OUT / "combined_2025_2026.csv", index=False)

    bootstrap, masks_2025, valid_2025 = _bootstrap(
        selected, outputs, (2025,), y, benefit, dates, currencies,
    )
    bootstrap["period"] = "2025"
    both_bootstrap, masks_both, valid_both = _bootstrap(
        selected, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    both_bootstrap["period"] = "2025_2026"
    pd.concat([bootstrap, both_bootstrap], ignore_index=True).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    circular = pd.concat([
        _circular_shift_audit(
            y, dates, currencies, valid_2025, masks_2025, "confirmation_2025",
        ),
        _circular_shift_audit(
            y, dates, currencies, valid_both, masks_both, "retrospective_2025_2026",
        ),
    ], ignore_index=True)
    circular.to_csv(OUT / "circular_shift_multiplicity.csv", index=False)

    breakdown = []
    for row in selected.itertuples(index=False):
        policy = Policy(
            int(row.history), float(row.strong), float(row.late),
            int(row.late_weekday), int(row.weekly_cap),
        )
        breakdown.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), policy,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)

    (OUT / "protocol.json").write_text(json.dumps({
        "next_rate_feature": False,
        "input_scores_frozen_from": str(SOURCE),
        "policy_selected_on": 2024,
        "confirmation": 2025, "audit": 2026,
        "n_policy_variants": len(policies()),
        "weekly_controller_is_causal": True,
        "quarter_history_reset": True,
        "pristine_holdout_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "corridor_freq_min", "quarter_frequency_min",
               "quarter_frequency_max", "clears_1p30_gate"]
    print("\n2024 SELECTED\n" + selected[[
        "candidate", "history", "strong", "late", "late_weekday", "weekly_cap",
        "frequency", "lift", "corridor_lift_min", "quarter_frequency_min",
    ]].to_string(index=False))
    print("\n2025 CONFIRMATION\n" + confirm[columns].to_string(index=False))
    print("\n2026 AUDIT\n" + audit[[
        "candidate", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "quarter_frequency_min", "passed_2025",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
