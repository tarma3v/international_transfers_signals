"""Walk-forward lift evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transfer_lift.features import feature_columns
from transfer_lift.models import default_model_names, make_model


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_walk_forward_folds(
    frame: pd.DataFrame,
    train_months: int = 36,
    test_months: int = 6,
    step_months: int = 6,
) -> list[Fold]:
    """Create chronological folds; all fitting data is strictly before test data."""
    min_date = pd.Timestamp(frame["date"].min()).normalize()
    max_date = pd.Timestamp(frame["date"].max()).normalize()
    test_start = min_date + pd.DateOffset(months=train_months)
    folds: list[Fold] = []
    while test_start <= max_date:
        train_start = test_start - pd.DateOffset(months=train_months)
        train_end = test_start - pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), max_date)
        if train_start < train_end and test_start < test_end:
            folds.append(Fold(train_start, train_end, test_start, test_end))
        test_start = test_start + pd.DateOffset(months=step_months)
    return folds


def _weekly_frequency(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    weeks = max((selected["date"].max() - selected["date"].min()).days / 7.0, 1.0)
    return float(len(selected) / weeks)


def _cluster_share(selected: pd.DataFrame, days: int = 3) -> float:
    if len(selected) <= 1:
        return 0.0
    sorted_days = selected.sort_values("date")["date"]
    gaps = sorted_days.diff().dt.days.dropna()
    return float((gaps <= days).mean()) if len(gaps) else 0.0


def _benefit_column_for_target(target_col: str) -> str:
    if target_col == "target_pub_fav":
        return "published_next_benefit_bps"
    return "benefit_bps"


def evaluate_predictions(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    top_rate: float = 0.15,
) -> pd.DataFrame:
    """Evaluate selected top-score days against random same-corridor/test-period baseline."""
    benefit_col = _benefit_column_for_target(target_col)
    rows: list[dict[str, object]] = []
    for corridor, part in frame.groupby("corridor", sort=True):
        part = part.dropna(subset=[target_col, score_col, benefit_col]).copy()
        if part.empty:
            continue
        limit = max(1, int(np.ceil(len(part) * top_rate)))
        selected = part.nlargest(limit, score_col)
        baseline_hit = float(part[target_col].mean())
        hit_rate = float(selected[target_col].mean())
        lift = hit_rate / baseline_hit if baseline_hit > 0 else np.nan
        rows.append(
            {
                "corridor": corridor,
                "n_test": int(len(part)),
                "n_signals": int(len(selected)),
                "hit_rate": hit_rate,
                "baseline_hit_rate": baseline_hit,
                "lift": lift,
                "mean_forward_benefit_bps": float(selected[benefit_col].mean()),
                "weekly_frequency": _weekly_frequency(selected),
                "cluster_share_3d": _cluster_share(selected),
            }
        )
    return pd.DataFrame(rows)


def _fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    target_col: str,
    features: list[str],
) -> pd.DataFrame:
    train = train.dropna(subset=[target_col]).copy()
    test = test.copy()
    if train[target_col].nunique() < 2:
        test["score"] = float(train[target_col].mean()) if len(train) else 0.0
        return test
    model = make_model(model_name)
    model.fit(train, target_col, features)
    test["score"] = model.score(test, features)
    return test


def benchmark_models(
    frame: pd.DataFrame,
    model_names: list[str] | None = None,
    target_col: str = "target_fav",
    top_rate: float = 0.15,
    train_months: int = 36,
    test_months: int = 6,
    step_months: int = 6,
) -> pd.DataFrame:
    """Run walk-forward benchmark and return model x corridor lift metrics."""
    models = model_names or default_model_names()
    features = feature_columns(frame, target_col=target_col)
    folds = make_walk_forward_folds(frame, train_months, test_months, step_months)
    if not folds:
        raise ValueError("not enough history for requested walk-forward folds")

    results: list[pd.DataFrame] = []
    for model_name in models:
        scored_folds: list[pd.DataFrame] = []
        for fold_id, fold in enumerate(folds):
            train = frame[(frame["date"] >= fold.train_start) & (frame["date"] <= fold.train_end)]
            test = frame[(frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)]
            scored = _fit_predict_fold(train, test, model_name, target_col, features)
            scored["fold"] = fold_id
            scored_folds.append(scored)
        scored_all = pd.concat(scored_folds, ignore_index=True)
        metrics = evaluate_predictions(scored_all, "score", target_col, top_rate=top_rate)
        metrics.insert(0, "model", model_name)
        results.append(metrics)
    return pd.concat(results, ignore_index=True).sort_values(["model", "corridor"]).reset_index(drop=True)


def summarize_overall(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compact model-level summary for deciding which candidates to keep."""
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            mean_lift=("lift", "mean"),
            min_lift=("lift", "min"),
            mean_hit_rate=("hit_rate", "mean"),
            mean_benefit_bps=("mean_forward_benefit_bps", "mean"),
            mean_weekly_frequency=("weekly_frequency", "mean"),
            mean_cluster_share_3d=("cluster_share_3d", "mean"),
        )
        .sort_values(["mean_lift", "mean_benefit_bps"], ascending=False)
        .reset_index(drop=True)
    )
