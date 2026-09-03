from __future__ import annotations

import numpy as np
import pandas as pd

from transfer_lift.evaluation import benchmark_models, evaluate_predictions, make_walk_forward_folds
from transfer_lift.features import build_dataset, feature_columns
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
        "target_pub_fav",
        "benefit_bps",
        "published_next_benefit_bps",
        "corridor",
        "date",
    }.issubset(frame.columns)
    safe_features = feature_columns(frame)
    assert "target_fav" not in safe_features
    assert "target_close" not in safe_features
    assert "benefit_bps" not in safe_features
    assert frame["target_fav"].isin([0.0, 1.0]).all()
    assert frame["target_pub_fav"].isin([0.0, 1.0]).all()


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
            "published_tomorrow_worse",
            "published_next_low",
            "level_low_percentile",
            "logistic_regression",
        ],
        top_rate=0.12,
        train_months=24,
        test_months=3,
        step_months=6,
    )
    assert set(metrics["model"]) == {
        "published_tomorrow_worse",
        "published_next_low",
        "level_low_percentile",
        "logistic_regression",
    }
    assert set(metrics["corridor"]) == {"TJS", "UZS", "KGS", "AMD", "KZT"}
    assert metrics["lift"].notna().all()
    assert (metrics["n_signals"] > 0).all()
