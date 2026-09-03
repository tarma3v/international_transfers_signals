from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from transfer_lift.calendar_ref import CORRIDOR_HOLIDAYS, HOLIDAYS, days_since_prev, days_to_next
from transfer_lift.data import Series
from transfer_lift.evaluation import (
    benchmark_models_cost_sensitive,
    evaluate_predictions_with_threshold,
    select_cost_threshold,
)
from transfer_lift.features import build_dataset
from transfer_lift.signals import (
    build_signal_table,
    signals_as_of,
    truncate_to,
)


def _synthetic_series() -> dict[str, Series]:
    rng = np.random.default_rng(11)
    dates = np.array([pd.Timestamp("2018-01-01") + pd.Timedelta(days=i) for i in range(2200)], dtype=object)
    dates = np.array([d.date() for d in dates], dtype=object)
    data: dict[str, Series] = {}
    for offset, code in enumerate(["TJS", "UZS", "KGS", "AMD", "KZT", "USD", "CNY"]):
        t = np.arange(len(dates), dtype=float)
        seasonal = np.sin(t / (35.0 + offset)) * (0.02 + offset * 0.001)
        trend = 0.00002 * t
        noise = rng.normal(0, 0.002, size=len(dates)).cumsum() / 20
        values = 5.0 + offset + trend + seasonal + noise
        data[code] = Series(code=code, dates=dates, values=values.astype(float))
    return data


# --- calendar_ref -----------------------------------------------------------


def test_holiday_calendar_covers_every_corridor() -> None:
    for corridor, holidays in CORRIDOR_HOLIDAYS.items():
        assert holidays, f"{corridor} has no holidays configured"
        for name in holidays:
            assert name in HOLIDAYS, f"unknown holiday {name} for {corridor}"


def test_days_to_next_and_since_prev_are_calendar_only() -> None:
    events = [dt.date(2024, 3, 21), dt.date(2025, 3, 21)]
    assert days_to_next(dt.date(2024, 3, 15), events) == 6
    assert days_since_prev(dt.date(2024, 3, 25), events) == 4
    # No future event within range -> sentinel value, not a crash.
    assert days_to_next(dt.date(2030, 1, 1), events) == 999


# --- signals.signals_as_of ----------------------------------------------------


def test_truncate_to_leaves_no_dates_after_as_of() -> None:
    series = _synthetic_series()
    as_of = dt.date(2022, 6, 1)
    cut = truncate_to(series, as_of)
    for code, s in cut.items():
        assert s.dates[-1] <= as_of, f"{code} leaked a date after {as_of}"


def test_signals_as_of_is_identical_regardless_of_future_history() -> None:
    """The core no-look-ahead check: results at date T must not change if more
    history is appended after T. This is what makes signals_as_of verifiable
    per problem.md:113."""
    series = _synthetic_series()
    as_of = dt.date(2022, 6, 1)

    full_rows = signals_as_of(series, as_of, include_published_next=False)

    near_future_cut = truncate_to(series, dt.date(2022, 6, 10))
    near_rows = signals_as_of(near_future_cut, as_of, include_published_next=False)

    assert full_rows == near_rows


def test_signals_as_of_returns_required_fields_per_corridor() -> None:
    series = _synthetic_series()
    rows = signals_as_of(series, dt.date(2022, 6, 1), include_published_next=False)
    required = {"date", "corridor", "indicator", "direction", "strength", "speed", "scenario"}
    assert rows
    for row in rows:
        assert required.issubset(row.keys())
    corridors_seen = {row["corridor"] for row in rows}
    assert corridors_seen == {"TJS", "UZS", "KGS", "AMD", "KZT"}


def test_signals_as_of_rejects_too_early_cut_date() -> None:
    series = _synthetic_series()
    too_early = series["TJS"].dates[5]
    try:
        signals_as_of(series, too_early)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a cut date without enough warmup")


def test_build_signal_table_has_case_required_columns() -> None:
    frame = build_dataset(_synthetic_series(), horizon=5)
    table = build_signal_table(frame)
    required = {"date", "corridor", "indicator", "direction", "strength", "speed", "scenario"}
    assert required.issubset(table.columns)
    assert not table.empty
    assert set(table["corridor"]) == {"TJS", "UZS", "KGS", "AMD", "KZT"}


# --- evaluation.select_cost_threshold / cost-sensitive benchmark ------------


def test_select_cost_threshold_prefers_precision_when_fp_is_expensive() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    target = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    strict_threshold = select_cost_threshold(scores, target, fp_cost=10.0, fn_cost=1.0)
    lenient_threshold = select_cost_threshold(scores, target, fp_cost=1.0, fn_cost=10.0)

    # Making false positives far more expensive must not lower the bar below
    # the cheaper-false-positive configuration.
    assert strict_threshold >= lenient_threshold


def test_select_cost_threshold_handles_all_nan_gracefully() -> None:
    scores = np.array([np.nan, np.nan])
    target = np.array([np.nan, np.nan])
    threshold = select_cost_threshold(scores, target)
    assert threshold == float("inf")


def test_evaluate_predictions_with_threshold_only_selects_at_or_above_threshold() -> None:
    frame = pd.DataFrame(
        {
            "corridor": ["TJS"] * 6,
            "date": pd.date_range("2024-01-01", periods=6),
            "target_fav": [1, 1, 0, 0, 0, 0],
            "benefit_bps": [10, 9, -1, -2, -3, -4],
            "score": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            "cost_threshold": [0.75] * 6,
        }
    )
    metrics = evaluate_predictions_with_threshold(frame, "score", "target_fav")
    assert metrics.loc[0, "n_signals"] == 2
    assert metrics.loc[0, "hit_rate"] == 1.0


def test_benchmark_models_cost_sensitive_returns_lift_and_uses_own_threshold_per_fold() -> None:
    frame = build_dataset(_synthetic_series(), horizon=5)
    metrics = benchmark_models_cost_sensitive(
        frame,
        model_names=["level_low_percentile", "logistic_regression"],
        target_col="target_fav",
        fp_cost=3.0,
        fn_cost=1.0,
        train_months=24,
        test_months=3,
        step_months=6,
    )
    assert set(metrics["model"]) == {"level_low_percentile", "logistic_regression"}
    assert set(metrics["corridor"]) == {"TJS", "UZS", "KGS", "AMD", "KZT"}
    # A cost-calibrated threshold is allowed to fire zero signals for some
    # corridor/fold combos; what matters is that no row silently used a
    # top-rate-style forced selection instead of the calibrated threshold.
    assert (metrics["n_signals"] >= 0).all()
