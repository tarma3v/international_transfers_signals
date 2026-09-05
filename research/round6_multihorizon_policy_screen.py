"""Packet-CU: tune only the causal decision policy for all required horizons."""
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
from research.round2_statistical_audit import _fired
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multihorizon_case_audit import corridor_period_adjusted_lift
from research.round6_uzbek_central_bank_models import _forward


OUT = Path("results/research/round6/multihorizon_policy_screen")
SOURCE = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
CANDIDATE = "geometry75_cba_consensus_basis25"


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    rate: float
    memory: int


def policy_grid():
    rolling = [
        PolicySpec("rolling", rate, window)
        for rate in (.18, .20, .22, .24, .26, .28, .30)
        for window in (10, 15, 20, 30, 40, 60, 90)
    ]
    exponential = [
        PolicySpec("exponential", rate, half_life)
        for rate in (.18, .20, .22, .24, .26, .28, .30)
        for half_life in (5, 10, 20, 40, 80)
    ]
    return rolling + exponential


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights)
    position = quantile * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, position), len(values) - 1)])


def exponential_fired(output, years, dates, currencies, y, rate, half_life):
    valid = np.zeros(len(y), dtype=bool)
    fired = np.zeros(len(y), dtype=bool)
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
            for row, score in zip(rows, scores):
                if len(history) >= 5:
                    ages = np.arange(len(history) - 1, -1, -1, dtype=float)
                    weights = np.exp(np.log(.5) * ages / half_life)
                    cutoff = _weighted_quantile(
                        np.asarray(history), weights, 1.0 - rate,
                    )
                    fired[row] = score >= cutoff
                history.append(float(score))
    return valid, fired


def fire(output, years, dates, currencies, y, spec):
    if spec.kind == "rolling":
        return _fired(
            output, years, dates, currencies, y,
            spec.rate, spec.memory, 0,
        )
    return exponential_fired(
        output, years, dates, currencies, y, spec.rate, spec.memory,
    )


def evaluate(output, years, spec, targets, forwards, dates, currencies):
    rows = []
    frequency = corridor_frequency_min = quarter_frequency_min = np.nan
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        valid, fired = fire(output, years, dates, currencies, y, spec)
        active = valid & fired
        lift, base, macro = corridor_period_adjusted_lift(
            y, valid, fired, currencies, dates, years,
        )
        if h == 5:
            frequency = rate_per_week(
                int(active.sum()), len(CORRIDORS), dates, valid,
            )
            corridor_frequency_min = min(
                rate_per_week(
                    int(np.sum(active & (currencies == currency))), 1,
                    dates, valid & (currencies == currency),
                )
                for currency in CORRIDORS
            )
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
            quarter_frequency_min = min(quarter_rates)
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
        "frequency": frequency,
        "corridor_frequency_min": corridor_frequency_min,
        "quarter_frequency_min": quarter_frequency_min,
    }
    return rows, summary


def future_score_check(output, dates, currencies, y):
    spec = PolicySpec("exponential", .22, 20)
    _valid, original = fire(output, (2025,), dates, currencies, y, spec)
    clone = {
        year: {key: np.asarray(value).copy() for key, value in item.items()}
        for year, item in output.items()
    }
    cutoff = dt.date(2025, 6, 30)
    rows = clone[2025]["test_idx"]
    future = np.asarray([dates[row] > cutoff for row in rows])
    clone[2025]["test_score"][future] = np.linspace(-1e6, 1e6, int(future.sum()))
    _valid, changed = fire(clone, (2025,), dates, currencies, y, spec)
    past = dates <= cutoff
    if not np.array_equal(original[past], changed[past]):
        raise AssertionError("future score changed a past exponential decision")


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
    for spec in policy_grid():
        detail, summary = evaluate(
            output, (2024,), spec, targets, forwards, dates, currencies,
        )
        policy_name = f"{spec.kind}_r{int(spec.rate*100):02d}_m{spec.memory:03d}"
        screen_rows.append({"policy": policy_name, **asdict(spec), **summary})
        screen_detail.extend({"policy": policy_name, **asdict(spec), **row} for row in detail)
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
    selected_row = feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean", "symmetric_benefit_min"],
        ascending=False,
    ).iloc[0]
    selected = PolicySpec(
        str(selected_row.kind), float(selected_row.rate), int(selected_row.memory),
    )
    default = PolicySpec("rolling", float(POLICY["rate"]), int(POLICY["rolling"]))

    later_rows = []
    later_summary = []
    for label, spec in (("default", default), ("selected", selected)):
        for period, years in (
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            rows, summary = evaluate(
                output, years, spec, targets, forwards, dates, currencies,
            )
            later_rows.extend({
                "candidate": label, "period": period, **asdict(spec), **row,
            } for row in rows)
            later_summary.append({
                "candidate": label, "period": period, **asdict(spec), **summary,
            })
    later = pd.DataFrame(later_rows)
    summaries = pd.DataFrame(later_summary)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    summaries.to_csv(OUT / "later_summary.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CU", "source_candidate": CANDIDATE,
        "selection_period": 2024, "selected_policy": asdict(selected),
        "default_policy": asdict(default), "candidate_count": len(screen),
        "selection_objective": "maximum worst official lift over h=1/3/5/10/20",
        "selection_constraints": "rate/corridor/quarter cadence and positive benefits",
        "exponential_quantile": "strictly previous scores, half-life in observations",
        "physical_future_score_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected:", selected)
    print("\nSCREEN TOP\n" + feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).head(20).to_string(index=False))
    print("\nLATER\n" + summaries.to_string(index=False))
    print("\nCOMBINED BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
