"""Packet-R threshold history reset at every scheduled model refit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from research.round2_statistical_audit import _bootstrap_all, _circular_shift_audit
from research.round5_features import load_round5_features


OUT = Path("results/research/round6/refit_threshold")
SOURCE = Path("results/research/round6/multiobjective_blend/outputs.pkl")
CANDIDATE = "stack50_benefit50"


@dataclass(frozen=True)
class ThresholdPolicy:
    rate: float
    window: int
    minimum_history: int


def policies():
    return [
        ThresholdPolicy(rate, window, minimum)
        for rate in (.18, .20, .22, .25, .30, .35)
        for window in (20, 40, 60, 120)
        for minimum in (5, 10, 20)
        if minimum <= window
    ]


def quarter_reset_fired(output, years, dates, currencies, y, policy):
    valid = np.zeros(len(y), dtype=bool)
    score = np.full(len(y), np.nan)
    for year in years:
        rows = np.asarray(output[year]["test_idx"], dtype=int)
        valid[rows] = np.isfinite(y[rows])
        score[rows] = np.asarray(output[year]["test_score"], dtype=float)
    fired = np.zeros(len(y), dtype=bool)
    for currency in CORRIDORS:
        rows = np.where(valid & (currencies == currency))[0]
        rows = rows[np.argsort(dates[rows])]
        history = []
        quarter_key = None
        for row in rows:
            day = dates[row]
            current_quarter = (day.year, (day.month - 1) // 3 + 1)
            if current_quarter != quarter_key:
                history = []
                quarter_key = current_quarter
            if len(history) >= policy.minimum_history:
                reference = np.asarray(history[-policy.window:], dtype=float)
                cutoff = float(np.quantile(reference, 1.0 - policy.rate))
                fired[row] = score[row] >= cutoff
            history.append(float(score[row]))
    return valid, fired, score


def summarize(output, years, dates, currencies, y, benefit, policy):
    valid, fired, _score = quarter_reset_fired(
        output, years, dates, currencies, y, policy,
    )
    active = valid & fired
    base = float(y[valid].mean())
    gains = benefit[active & ~np.isnan(benefit)]
    corridor_lift, corridor_freq = [], []
    for currency in CORRIDORS:
        scope = valid & (currencies == currency)
        signals = active & (currencies == currency)
        corridor_lift.append(float(y[signals].mean() / y[scope].mean()))
        corridor_freq.append(rate_per_week(int(signals.sum()), 1, dates, scope))
    quarter_lift, quarter_freq, year_lift, year_freq = [], [], [], []
    for year in years:
        scope = valid & np.asarray([day.year == year for day in dates])
        signals = active & scope
        year_lift.append(float(y[signals].mean() / y[scope].mean()))
        year_freq.append(rate_per_week(int(signals.sum()), len(CORRIDORS), dates, scope))
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                day.year == year and (day.month - 1) // 3 + 1 == quarter for day in dates
            ])
            if not scope.any():
                continue
            signals = active & scope
            quarter_freq.append(rate_per_week(
                int(signals.sum()), len(CORRIDORS), dates, scope,
            ))
            quarter_lift.append(
                float(y[signals].mean() / y[scope].mean()) if signals.any() else np.nan
            )
    return {
        "n": int(active.sum()),
        "frequency": rate_per_week(int(active.sum()), len(CORRIDORS), dates, valid),
        "hit_rate": float(y[active].mean()), "base_rate": base,
        "lift": float(y[active].mean() / base),
        "forward_benefit_bps": float(np.mean(gains)) if len(gains) else np.nan,
        "corridor_lift_min": float(min(corridor_lift)),
        "corridor_freq_min": float(min(corridor_freq)),
        "corridor_freq_max": float(max(corridor_freq)),
        "quarter_lift_min": float(np.nanmin(quarter_lift)),
        "quarter_frequency_min": float(min(quarter_freq)),
        "quarter_frequency_max": float(max(quarter_freq)),
        "macro_year_lift": float(np.mean(year_lift)),
        "year_lift_min": float(min(year_lift)),
        "year_frequency_min": float(min(year_freq)),
        "year_frequency_max": float(max(year_freq)),
    }


def breakdown(output, years, dates, currencies, y, benefit, policy):
    valid, fired, _score = quarter_reset_fired(
        output, years, dates, currencies, y, policy,
    )
    groups = [("overall", "all", valid, len(CORRIDORS))]
    for year in years:
        groups.append((
            "year", str(year), valid & np.asarray([d.year == year for d in dates]),
            len(CORRIDORS),
        ))
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                d.year == year and (d.month - 1) // 3 + 1 == quarter for d in dates
            ])
            if scope.any():
                groups.append(("quarter", f"{year}Q{quarter}", scope, len(CORRIDORS)))
    for currency in CORRIDORS:
        groups.append(("currency", currency, valid & (currencies == currency), 1))
    rows = []
    for kind, group, scope, n_corridors in groups:
        active = scope & fired
        gains = benefit[active & ~np.isnan(benefit)]
        rows.append({
            "breakdown": kind, "group": group, "n_scope": int(scope.sum()),
            "n_signals": int(active.sum()),
            "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
            "base_rate": float(y[scope].mean()),
            "hit_rate": float(y[active].mean()) if active.any() else np.nan,
            "lift": float(y[active].mean() / y[scope].mean()) if active.any() else np.nan,
            "forward_benefit_bps": float(np.mean(gains)) if len(gains) else np.nan,
        })
    return rows


def future_score_check(output, dates, currencies, y):
    policy = ThresholdPolicy(.22, 60, 10)
    _valid, original, _score = quarter_reset_fired(
        output, (2025,), dates, currencies, y, policy,
    )
    clone = {year: {key: np.asarray(value).copy() for key, value in part.items()}
             for year, part in output.items()}
    cut = dt.date(2025, 6, 30)
    rows = clone[2025]["test_idx"]
    future = np.asarray([dates[row] > cut for row in rows])
    clone[2025]["test_score"][future] = np.linspace(-1000, 1000, int(future.sum()))
    _valid, changed, _score = quarter_reset_fired(
        clone, (2025,), dates, currencies, y, policy,
    )
    past = np.asarray([day <= cut for day in dates])
    if not np.array_equal(original[past], changed[past]):
        raise AssertionError("future score changed a past quarter-reset decision")


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
    with SOURCE.open("rb") as handle:
        output = pickle.load(handle)[CANDIDATE]
    future_score_check(output, dates, currencies, y)

    screen_rows = []
    for policy in policies():
        item = summarize(output, (2024,), dates, currencies, y, benefit, policy)
        item.update(asdict(policy))
        screen_rows.append(item)
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(OUT / "screen_2024.csv", index=False)
    feasible = screen[
        screen.frequency.between(1.00, 2.00)
        & screen.corridor_freq_min.ge(.80)
        & screen.quarter_frequency_min.ge(.70)
        & screen.forward_benefit_bps.gt(0)
    ].copy()
    if feasible.empty:
        raise RuntimeError("no feasible packet-R threshold on 2024")
    feasible["robustness"] = feasible[["lift", "corridor_lift_min"]].min(axis=1)
    chosen = feasible.sort_values(
        ["robustness", "lift", "quarter_frequency_min"], ascending=False,
    ).iloc[0]
    chosen.to_frame().T.to_csv(OUT / "selected_2024.csv", index=False)
    policy = ThresholdPolicy(
        float(chosen.rate), int(chosen.window), int(chosen.minimum_history),
    )

    rows = []
    for period, years in (
        ("screen_2024", (2024,)),
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        item = summarize(output, years, dates, currencies, y, benefit, policy)
        item.update({"period": period, **asdict(policy)})
        rows.append(item)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(breakdown(
        output, (2025, 2026), dates, currencies, y, benefit, policy,
    )).to_csv(OUT / "breakdown_2025_2026.csv", index=False)

    valid, fired, _score = quarter_reset_fired(
        output, (2025, 2026), dates, currencies, y, policy,
    )
    masks = {"refit_threshold": fired}
    draws = _bootstrap_all(y, benefit, dates, valid, masks)["refit_threshold"]
    lift_draws = draws["lift"][np.isfinite(draws["lift"])]
    benefit_draws = draws["benefit"][np.isfinite(draws["benefit"])]
    pd.DataFrame([{
        "lift_ci_low": float(np.quantile(lift_draws, .025)),
        "lift_ci_high": float(np.quantile(lift_draws, .975)),
        "p_lift_le_1": float((np.sum(lift_draws <= 1) + 1) / (len(lift_draws) + 1)),
        "benefit_ci_low": float(np.quantile(benefit_draws, .025)),
        "benefit_ci_high": float(np.quantile(benefit_draws, .975)),
    }]).to_csv(OUT / "block_bootstrap.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "retrospective_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "candidate": CANDIDATE,
        "policy": asdict(policy),
        "history_reset": "calendar quarter, aligned with model refit",
        "current_score_appended_after_decision": True,
        "physical_future_score_corruption_check": True,
        "selection_period": 2024,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "corridor_lift_min", "quarter_frequency_min",
        "quarter_frequency_max", "quarter_lift_min",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
