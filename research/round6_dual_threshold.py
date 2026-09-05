"""Packet-S sparse quarter-reset fallback on the multi-objective score."""
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
from research.round2_statistical_audit import _bootstrap_all, _circular_shift_audit, _fired
from research.round5_features import load_round5_features
from research.round6_refit_threshold import ThresholdPolicy, quarter_reset_fired


OUT = Path("results/research/round6/dual_threshold")
SOURCE = Path("results/research/round6/multiobjective_blend/outputs.pkl")
CANDIDATE = "stack50_benefit50"
RESET_POLICY = ThresholdPolicy(.35, 20, 10)


@dataclass(frozen=True)
class DualPolicy:
    lookback_days: int
    primary_ceiling: float
    emitted_floor: float
    fallback_weekday: int


def policies():
    return [
        DualPolicy(days, ceiling, floor, weekday)
        for days in (14, 21, 28, 42)
        for ceiling in (.50, .75, 1.00)
        for floor in (.75, 1.00, 1.25)
        for weekday in (0, 2, 3)
    ]


def dual_fired(output, target_years, dates, currencies, y, policy):
    history_years = tuple(range(2024, max(target_years) + 1))
    valid, primary = _fired(
        output, history_years, dates, currencies, y, .22, 60, 0,
    )
    reset_valid, reset, _score = quarter_reset_fired(
        output, history_years, dates, currencies, y, RESET_POLICY,
    )
    if not np.array_equal(valid, reset_valid):
        raise AssertionError("ordinary/reset evaluation rows differ")
    fired = np.zeros(len(y), dtype=bool)
    source = np.full(len(y), "", dtype=object)
    for currency in CORRIDORS:
        rows = np.where(valid & (currencies == currency))[0]
        rows = rows[np.argsort(dates[rows])]
        primary_dates: list[dt.date] = []
        emitted_dates: list[dt.date] = []
        week_key = None
        fallback_count = 0
        for row in rows:
            day = dates[row]
            current_week = tuple(day.isocalendar()[:2])
            if current_week != week_key:
                week_key = current_week
                fallback_count = 0
            left = day - dt.timedelta(days=policy.lookback_days)
            primary_rate = sum(left <= value < day for value in primary_dates) * 7.0 / policy.lookback_days
            emitted_rate = sum(left <= value < day for value in emitted_dates) * 7.0 / policy.lookback_days
            if primary[row]:
                fired[row] = True
                source[row] = "primary"
            elif (
                fallback_count < 1
                and day.weekday() >= policy.fallback_weekday
                and primary_rate < policy.primary_ceiling
                and emitted_rate < policy.emitted_floor
                and reset[row]
            ):
                fired[row] = True
                source[row] = "reset"
                fallback_count += 1
            if primary[row]:
                primary_dates.append(day)
            if fired[row]:
                emitted_dates.append(day)
    target_valid = valid & np.asarray([day.year in target_years for day in dates])
    return target_valid, fired & target_valid, source


def summarize(output, years, dates, currencies, y, benefit, policy):
    valid, fired, source = dual_fired(output, years, dates, currencies, y, policy)
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
        "primary_n": int(np.sum(active & (source == "primary"))),
        "reset_n": int(np.sum(active & (source == "reset"))),
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
    valid, fired, source = dual_fired(output, years, dates, currencies, y, policy)
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
    result = []
    for kind, group, scope, n_corridors in groups:
        active = scope & fired
        gains = benefit[active & ~np.isnan(benefit)]
        result.append({
            "breakdown": kind, "group": group, "n_scope": int(scope.sum()),
            "n_signals": int(active.sum()),
            "primary_n": int(np.sum(active & (source == "primary"))),
            "reset_n": int(np.sum(active & (source == "reset"))),
            "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
            "base_rate": float(y[scope].mean()),
            "hit_rate": float(y[active].mean()) if active.any() else np.nan,
            "lift": float(y[active].mean() / y[scope].mean()) if active.any() else np.nan,
            "forward_benefit_bps": float(np.mean(gains)) if len(gains) else np.nan,
        })
    return result


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
        & screen.quarter_frequency_min.ge(.90)
        & screen.forward_benefit_bps.gt(0)
    ].copy()
    if feasible.empty:
        raise RuntimeError("no feasible packet-S dual threshold on 2024")
    feasible["robustness"] = feasible[["lift", "corridor_lift_min"]].min(axis=1)
    chosen = feasible.sort_values(
        ["robustness", "lift", "quarter_frequency_min"], ascending=False,
    ).iloc[0]
    chosen.to_frame().T.to_csv(OUT / "selected_2024.csv", index=False)
    policy = DualPolicy(
        int(chosen.lookback_days), float(chosen.primary_ceiling),
        float(chosen.emitted_floor), int(chosen.fallback_weekday),
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

    valid, fired, _source = dual_fired(
        output, (2025, 2026), dates, currencies, y, policy,
    )
    masks = {"dual_threshold": fired}
    draws = _bootstrap_all(y, benefit, dates, valid, masks)["dual_threshold"]
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
        "primary": {"rate": .22, "rolling": 60},
        "reset_fallback": asdict(RESET_POLICY),
        "dual_policy": asdict(policy),
        "fallback_cap": "one per currency ISO week; primary uncapped",
        "selection_period": 2024,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "period", "frequency", "lift", "forward_benefit_bps", "primary_n",
        "reset_n", "corridor_freq_min", "corridor_lift_min",
        "quarter_frequency_min", "quarter_frequency_max", "quarter_lift_min",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
