"""Packet-CW: causal first/second-alert weekly policy for the incumbent score."""
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
from ml.targets import HORIZONS, build_targets
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multihorizon_case_audit import corridor_period_adjusted_lift
from research.round6_multihorizon_policy_screen import evaluate as evaluate_default
from research.round6_uzbek_central_bank_models import _forward


OUT = Path("results/research/round6/weekly_confidence_policy")
SOURCE = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
CANDIDATE = "geometry75_cba_consensus_basis25"


@dataclass(frozen=True)
class WeeklyPolicy:
    history: int
    first: float
    second: float
    fallback: float
    fallback_weekday: int
    cap: int = 2


def policies():
    return [
        WeeklyPolicy(history, first, second, fallback, weekday)
        for history in (20, 40, 60)
        for first in (.70, .75, .80, .85)
        for second in (.88, .92, .96)
        for fallback in (.50, .60, .70)
        for weekday in (3, 4)
        if fallback < first < second
    ]


def _percentile(history, value):
    ordered = np.sort(np.asarray(history, dtype=float))
    return float(np.searchsorted(ordered, value, side="right") / len(ordered))


def weekly_fired(output, years, dates, currencies, y, policy):
    valid = np.zeros(len(y), dtype=bool)
    fired = np.zeros(len(y), dtype=bool)
    ranks = np.full(len(y), np.nan)
    for year in years:
        if year not in output:
            continue
        part = output[year]
        calibration = np.asarray(part["calib_idx"], dtype=int)
        test = np.asarray(part["test_idx"], dtype=int)
        valid[test] = np.isfinite(y[test])
        for currency in CORRIDORS:
            ca_mask = currencies[calibration] == currency
            te_mask = currencies[test] == currency
            ca_order = np.argsort(dates[calibration[ca_mask]])
            te_order = np.argsort(dates[test[te_mask]])
            history = list(np.asarray(part["calib_score"])[ca_mask][ca_order])
            rows = test[te_mask][te_order]
            scores = np.asarray(part["test_score"])[te_mask][te_order]
            week = None
            count = 0
            for row, score in zip(rows, scores):
                day = dates[row]
                week_key = tuple(day.isocalendar()[:2])
                if week_key != week:
                    week = week_key
                    count = 0
                if history:
                    rank = _percentile(history[-policy.history:], score)
                    ranks[row] = rank
                    threshold = policy.first if count == 0 else policy.second
                    ordinary = count < policy.cap and rank >= threshold
                    fallback = (
                        count == 0 and day.weekday() >= policy.fallback_weekday
                        and rank >= policy.fallback
                    )
                    fired[row] = ordinary or fallback
                    if fired[row]:
                        count += 1
                history.append(float(score))
    return valid, fired, ranks


def evaluate(output, years, policy, targets, forwards, dates, currencies):
    rows = []
    summary_fields = {}
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        valid, fired, _ranks = weekly_fired(
            output, years, dates, currencies, y, policy,
        )
        active = valid & fired
        lift, base, macro = corridor_period_adjusted_lift(
            y, valid, fired, currencies, dates, years,
        )
        if h == 5:
            corridor_rates = [
                rate_per_week(
                    int(np.sum(active & (currencies == currency))), 1,
                    dates, valid & (currencies == currency),
                )
                for currency in CORRIDORS
            ]
            quarter_rates = []
            for year in years:
                for quarter in range(1, 5):
                    scope = valid & np.asarray([
                        day.year == year and (day.month - 1) // 3 + 1 == quarter
                        for day in dates
                    ])
                    if scope.any():
                        quarter_rates.append(rate_per_week(
                            int(np.sum(active & scope)), len(CORRIDORS), dates, scope,
                        ))
            summary_fields = {
                "frequency": rate_per_week(
                    int(active.sum()), len(CORRIDORS), dates, valid,
                ),
                "corridor_frequency_min": min(corridor_rates),
                "corridor_frequency_max": max(corridor_rates),
                "quarter_frequency_min": min(quarter_rates),
                "quarter_frequency_max": max(quarter_rates),
            }
        rows.append({
            "horizon": h, "case_lift": lift,
            "matched_random_day_rate": base,
            "macro_corridor_year_lift": macro,
            "symmetric_benefit_bps": float(np.nanmean(
                targets[f"benefit_h{h}"][active]
            )),
            "future_only_benefit_bps": float(np.nanmean(forwards[h][active])),
            "n_signals": int(active.sum()),
        })
    summary = {
        "horizon_lift_min": min(row["case_lift"] for row in rows),
        "horizon_lift_mean": float(np.mean([row["case_lift"] for row in rows])),
        "symmetric_benefit_min": min(row["symmetric_benefit_bps"] for row in rows),
        "future_benefit_min": min(row["future_only_benefit_bps"] for row in rows),
        **summary_fields,
    }
    return rows, summary


def future_score_check(output, dates, currencies, y):
    policy = WeeklyPolicy(40, .75, .92, .60, 3)
    _valid, original, _ranks = weekly_fired(
        output, (2025,), dates, currencies, y, policy,
    )
    clone = {
        year: {key: np.asarray(value).copy() for key, value in part.items()}
        for year, part in output.items()
    }
    cutoff = dt.date(2025, 6, 30)
    rows = clone[2025]["test_idx"]
    future = np.asarray([dates[row] > cutoff for row in rows])
    clone[2025]["test_score"][future] = np.linspace(-1e6, 1e6, int(future.sum()))
    _valid, changed, _ranks = weekly_fired(
        clone, (2025,), dates, currencies, y, policy,
    )
    if not np.array_equal(original[dates <= cutoff], changed[dates <= cutoff]):
        raise AssertionError("future score changed a past weekly decision")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    with SOURCE.open("rb") as handle:
        output = pickle.load(handle)[CANDIDATE]
    future_score_check(output, dates, currencies, targets["fav_h5"])

    screen_rows = []
    screen_detail = []
    for policy in policies():
        detail, summary = evaluate(
            output, (2024,), policy, targets, forwards, dates, currencies,
        )
        name = (
            f"h{policy.history}_f{int(policy.first*100)}_"
            f"s{int(policy.second*100)}_b{int(policy.fallback*100)}_"
            f"d{policy.fallback_weekday}"
        )
        screen_rows.append({"policy": name, **asdict(policy), **summary})
        screen_detail.extend({"policy": name, **asdict(policy), **row} for row in detail)
    screen = pd.DataFrame(screen_rows)
    detail = pd.DataFrame(screen_detail)
    screen.to_csv(OUT / "screen_2024_summary.csv", index=False)
    detail.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    feasible = screen[
        screen.frequency.between(1.0, 2.0)
        & screen.corridor_frequency_min.ge(.80)
        & screen.quarter_frequency_min.ge(.70)
        & screen.symmetric_benefit_min.gt(0)
        & screen.future_benefit_min.gt(0)
    ]
    chosen_row = feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "symmetric_benefit_min"],
        ascending=False,
    ).iloc[0]
    chosen = WeeklyPolicy(
        int(chosen_row["history"]), float(chosen_row["first"]),
        float(chosen_row["second"]), float(chosen_row["fallback"]),
        int(chosen_row["fallback_weekday"]), int(chosen_row["cap"]),
    )

    later_rows = []
    later_summaries = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        rows, summary = evaluate(
            output, years, chosen, targets, forwards, dates, currencies,
        )
        later_rows.extend({"period": period, **asdict(chosen), **row} for row in rows)
        later_summaries.append({"period": period, **asdict(chosen), **summary})
    later = pd.DataFrame(later_rows)
    summaries = pd.DataFrame(later_summaries)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    summaries.to_csv(OUT / "later_summary.csv", index=False)
    # Compute the exact default numbers beside the selected weekly policy.
    from research.round6_multihorizon_policy_screen import PolicySpec
    _rows, default_summary = evaluate_default(
        output, (2025, 2026), PolicySpec("rolling", .22, 20),
        targets, forwards, dates, currencies,
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CW", "source_candidate": CANDIDATE,
        "selection_period": 2024, "selected_policy": asdict(chosen),
        "candidate_count": len(screen), "weekly_cap": 2,
        "current_score_enters_history_after_decision": True,
        "selection_objective": "maximum worst official lift over all five horizons with cadence and benefit constraints",
        "physical_future_score_corruption_check": True,
        "default_combined_summary": default_summary,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected:", chosen)
    print("\nSCREEN TOP\n" + feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).head(20).to_string(index=False))
    print("\nLATER\n" + summaries.to_string(index=False))
    print("\nCOMBINED BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
