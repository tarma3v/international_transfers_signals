from __future__ import annotations

import numpy as np
import pandas as pd

from transfer_lift.evaluation import (
    benchmark_models,
    evaluate_predictions,
    make_walk_forward_folds,
    purge_label_overlap,
)
from transfer_lift.features import build_dataset, feature_columns, target_local_minimum
from transfer_lift.data import Series


def _synthetic_series() -> dict[str, Series]:
    rng = np.random.default_rng(7)
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


def test_evaluate_predictions_lift_is_above_random_when_score_knows_target() -> None:
    frame = pd.DataFrame(
        {
            "corridor": ["TJS"] * 10,
            "date": pd.date_range("2024-01-01", periods=10),
            "target_fav": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "benefit_bps": [10, 9, 8, -1, -2, -3, -4, -5, -6, -7],
            "score": [0.99, 0.95, 0.9, 0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.01],
        }
    )
    metrics = evaluate_predictions(frame, "score", "target_fav", top_rate=0.3)
    assert metrics.loc[0, "hit_rate"] == 1.0
    assert metrics.loc[0, "baseline_hit_rate"] == 0.3
    assert metrics.loc[0, "lift"] > 3.0


def test_feature_builder_has_targets_and_no_target_columns_in_features() -> None:
    frame = build_dataset(_synthetic_series(), horizon=5)
    assert {
        "target_fav",
        "target_close",
        "target_local_min",
        "benefit_bps",
        "symmetric_benefit_bps",
        "corridor",
        "date",
    }.issubset(frame.columns)
    safe_features = feature_columns(frame)
    assert "target_fav" not in safe_features
    assert "target_close" not in safe_features
    assert "target_local_min" not in safe_features
    assert "benefit_bps" not in safe_features
    assert "symmetric_benefit_bps" not in safe_features
    assert frame["target_fav"].isin([0.0, 1.0]).all()
    assert frame["target_local_min"].isin([0.0, 1.0]).all()


def test_target_local_minimum_matches_symmetric_window_definition() -> None:
    values = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64)
    # idx=4 (value 5.0) is the global minimum: symmetric +-2 window is fully inside bounds.
    assert target_local_minimum(values, idx=4, horizon=2) == 1.0
    # idx=3 (value 6.0) is not a local minimum of its own +-2 window (5.0 is lower, at idx=4).
    assert target_local_minimum(values, idx=3, horizon=2) == 0.0
    # Out-of-bounds window (not enough history on the left) must return None, not leak partial data.
    assert target_local_minimum(values, idx=1, horizon=2) is None
    assert target_local_minimum(values, idx=8, horizon=2) is None


def test_purge_label_overlap_removes_last_h_rows_per_corridor() -> None:
    frame = pd.DataFrame(
        {
            "corridor": ["TJS"] * 5 + ["KZT"] * 5,
            "date": list(pd.date_range("2024-01-01", periods=5)) * 2,
        }
    )

    purged = purge_label_overlap(frame, horizon=2)

    assert purged.groupby("corridor").size().to_dict() == {"KZT": 3, "TJS": 3}
    assert purged.groupby("corridor")["date"].max().eq(pd.Timestamp("2024-01-03")).all()


def test_evaluate_predictions_selects_top_rate_inside_each_fold() -> None:
    frame = pd.DataFrame(
        {
            "corridor": ["TJS"] * 8,
            "fold": [0] * 4 + [1] * 4,
            "date": pd.date_range("2024-01-01", periods=8),
            "target_fav": [1, 0, 0, 0, 1, 0, 0, 0],
            "benefit_bps": [10, 0, 0, 0, 10, 0, 0, 0],
            "score": [0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05, 0.01],
        }
    )

    metrics = evaluate_predictions(frame, "score", "target_fav", top_rate=0.25)

    assert metrics.loc[0, "n_signals"] == 2
    assert metrics.loc[0, "hit_rate"] == 1.0


def test_walk_forward_folds_do_not_overlap() -> None:
    frame = build_dataset(_synthetic_series(), horizon=5)
    folds = make_walk_forward_folds(frame, train_months=24, test_months=3, step_months=3)
    assert folds
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_benchmark_models_returns_lift_for_rule_and_ml_models() -> None:
    frame = build_dataset(_synthetic_series(), horizon=5)
    metrics = benchmark_models(
        frame,
        model_names=[
            "momentum_down",
            "level_low_percentile",
            "logistic_regression",
        ],
        top_rate=0.12,
        train_months=24,
        test_months=3,
        step_months=6,
    )
    assert set(metrics["model"]) == {
        "momentum_down",
        "level_low_percentile",
        "logistic_regression",
    }
    assert set(metrics["corridor"]) == {"TJS", "UZS", "KGS", "AMD", "KZT"}
    assert metrics["lift"].notna().all()
    assert (metrics["n_signals"] > 0).all()


def test_benchmark_models_on_symmetric_local_minimum_target() -> None:
    frame = build_dataset(_synthetic_series(), horizon=5)
    metrics = benchmark_models(
        frame,
        model_names=["level_low_percentile", "logistic_regression"],
        target_col="target_local_min",
        top_rate=0.12,
        train_months=24,
        test_months=3,
        step_months=6,
    )
    assert set(metrics["model"]) == {"level_low_percentile", "logistic_regression"}
    assert metrics["lift"].notna().all()
    assert (metrics["n_signals"] > 0).all()
