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


def purge_label_overlap(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Drop each corridor's last h rows whose labels overlap the next test period."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0 or train.empty:
        return train.copy()
    ordered = train.sort_values(["corridor", "date"])
    grouped = ordered.groupby("corridor", sort=False)
    position = grouped.cumcount()
    group_size = grouped["corridor"].transform("size")
    return ordered.loc[position < group_size - horizon].reset_index(drop=True)


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
    if target_col == "target_local_min":
        return "symmetric_benefit_bps"
    return "benefit_bps"


def select_cost_threshold(
    scores: np.ndarray,
    target: np.ndarray,
    fp_cost: float = 3.0,
    fn_cost: float = 1.0,
    min_signal_rate: float = 0.0,
) -> float:
    """Pick a score threshold that minimizes asymmetric misclassification cost.

    The case is explicit that error cost is NOT symmetric (problem.md:125):
    "«Сказали переводить — курс ушёл ещё ниже» дороже, чем пропущенный удачный
    день... Пороги и метрики строятся с учётом этой асимметрии, а не
    оптимизируют симметричную точность." A false positive here means we told
    the client "favourable now" and it was not true (`target == 0`); a false
    negative means we stayed silent on a day that was in fact favourable
    (`target == 1`). By default `fp_cost > fn_cost`, matching that a bad
    signal is strictly worse than a missed good day.

    `min_signal_rate` optionally excludes thresholds that would select too
    few days to be useful for a pilot (case's own tension between frequency
    being too high or too low, problem.md:160).

    Must be called on a fold's TRAIN scores/targets, never on test, otherwise
    the threshold itself becomes a source of test-set leakage.
    """
    valid = ~np.isnan(scores) & ~np.isnan(target)
    scores, target = scores[valid], target[valid].astype(bool)
    if len(scores) == 0:
        return float("inf")

    candidates = np.unique(scores)
    best_threshold = float(candidates.max()) + 1.0  # default: select nothing
    best_cost = float("inf")
    n = len(scores)
    for threshold in candidates:
        fired = scores >= threshold
        rate = float(fired.mean())
        if rate < min_signal_rate:
            continue
        false_positive = int((fired & ~target).sum())
        false_negative = int((~fired & target).sum())
        cost = fp_cost * false_positive + fn_cost * false_negative
        if cost < best_cost - 1e-12 or (
            abs(cost - best_cost) <= 1e-12 and rate > 0 and threshold < best_threshold
        ):
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold


def _metric_row(
    corridor: str,
    part: pd.DataFrame,
    selected: pd.DataFrame,
    target_col: str,
    benefit_col: str,
) -> dict[str, object]:
    """Calculate the shared metric set for one corridor."""
    baseline_hit = float(part[target_col].mean())
    n_signals = len(selected)
    hit_rate = float(selected[target_col].mean()) if n_signals else np.nan
    lift = hit_rate / baseline_hit if baseline_hit > 0 and n_signals else np.nan
    return {
        "corridor": corridor,
        "n_test": len(part),
        "n_signals": n_signals,
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline_hit,
        "lift": lift,
        "mean_forward_benefit_bps": (
            float(selected[benefit_col].mean()) if n_signals else np.nan
        ),
        "weekly_frequency": _weekly_frequency(selected),
        "cluster_share_3d": _cluster_share(selected),
    }


def _select_top_rate(part: pd.DataFrame, score_col: str, top_rate: float) -> pd.DataFrame:
    """Select the requested score share independently inside every test fold."""
    group_columns = ["fold"] if "fold" in part.columns else []
    groups = part.groupby(group_columns, sort=False) if group_columns else [(None, part)]
    selected: list[pd.DataFrame] = []
    for _, group in groups:
        limit = max(1, int(np.ceil(len(group) * top_rate)))
        selected.append(group.nlargest(limit, score_col))
    return pd.concat(selected, ignore_index=False)


def evaluate_predictions(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    top_rate: float = 0.15,
) -> pd.DataFrame:
    """Evaluate top-score days selected separately in each out-of-time fold."""
    benefit_col = _benefit_column_for_target(target_col)
    rows: list[dict[str, object]] = []
    for corridor, part in frame.groupby("corridor", sort=True):
        part = part.dropna(subset=[target_col, score_col, benefit_col])
        if part.empty:
            continue
        selected = _select_top_rate(part, score_col, top_rate)
        rows.append(_metric_row(corridor, part, selected, target_col, benefit_col))
    return pd.DataFrame(rows)


def evaluate_predictions_with_threshold(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    threshold_col: str = "cost_threshold",
) -> pd.DataFrame:
    """Same metrics as `evaluate_predictions`, but signals are `score >= threshold`
    instead of a fixed top-K share. `threshold_col` must already hold, per row, the
    threshold chosen on that row's TRAIN fold (see `benchmark_models_cost_sensitive`).
    """
    benefit_col = _benefit_column_for_target(target_col)
    rows: list[dict[str, object]] = []
    required = [target_col, score_col, benefit_col, threshold_col]
    for corridor, part in frame.groupby("corridor", sort=True):
        part = part.dropna(subset=required)
        if part.empty:
            continue
        selected = part[part[score_col] >= part[threshold_col]]
        rows.append(_metric_row(corridor, part, selected, target_col, benefit_col))
    return pd.DataFrame(rows)


def _fit_and_score(
    train: pd.DataFrame,
    frames: list[pd.DataFrame],
    model_name: str,
    target_col: str,
    features: list[str],
) -> list[pd.DataFrame]:
    """Fit once and score one or more frames with the same fitted model."""
    train = train.dropna(subset=[target_col])
    scored_frames = [frame.copy() for frame in frames]
    if train[target_col].nunique() < 2:
        constant_score = float(train[target_col].mean()) if len(train) else 0.0
        for scored in scored_frames:
            scored["score"] = constant_score
        return scored_frames

    model = make_model(model_name).fit(train, target_col, features)
    for scored in scored_frames:
        scored["score"] = model.score(scored, features)
    return scored_frames


def _fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    target_col: str,
    features: list[str],
) -> pd.DataFrame:
    return _fit_and_score(train, [test], model_name, target_col, features)[0]


def benchmark_models(
    frame: pd.DataFrame,
    model_names: list[str] | None = None,
    target_col: str = "target_fav",
    top_rate: float = 0.15,
    train_months: int = 36,
    test_months: int = 6,
    step_months: int = 6,
    horizon: int = 5,
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
            train = purge_label_overlap(train, horizon)
            test = frame[(frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)]
            scored = _fit_predict_fold(train, test, model_name, target_col, features)
            scored["fold"] = fold_id
            scored_folds.append(scored)
        scored_all = pd.concat(scored_folds, ignore_index=True)
        metrics = evaluate_predictions(scored_all, "score", target_col, top_rate=top_rate)
        metrics.insert(0, "model", model_name)
        results.append(metrics)
    return pd.concat(results, ignore_index=True).sort_values(["model", "corridor"]).reset_index(drop=True)


def benchmark_models_cost_sensitive(
    frame: pd.DataFrame,
    model_names: list[str] | None = None,
    target_col: str = "target_fav",
    fp_cost: float = 3.0,
    fn_cost: float = 1.0,
    min_signal_rate: float = 0.0,
    train_months: int = 36,
    test_months: int = 6,
    step_months: int = 6,
    horizon: int = 5,
) -> pd.DataFrame:
    """Same walk-forward scheme as `benchmark_models`, but instead of a fixed
    `top_rate`, each (model, corridor, fold) gets its own threshold selected on
    TRAIN scores via `select_cost_threshold`, then applied unmodified to TEST.

    Directly implements problem.md:46 ("порог, учитывающим асимметричную цену
    ошибки") and problem.md:50 ("откалибровать окна и пороги отдельно для
    каждого коридора, поскольку их волатильность различается") together: the
    threshold is asymmetric-cost-aware AND per-corridor, because train/test are
    already split by corridor inside `evaluate_predictions_with_threshold`.
    """
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
            train = purge_label_overlap(train, horizon)
            test = frame[(frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)]
            scored_train, scored_test = _fit_and_score(
                train, [train, test], model_name, target_col, features
            )

            thresholds: dict[str, float] = {}
            for corridor, part in scored_train.groupby("corridor"):
                thresholds[corridor] = select_cost_threshold(
                    part["score"].to_numpy(),
                    part[target_col].to_numpy(),
                    fp_cost=fp_cost,
                    fn_cost=fn_cost,
                    min_signal_rate=min_signal_rate,
                )
            scored_test = scored_test.copy()
            scored_test["cost_threshold"] = scored_test["corridor"].map(thresholds)
            scored_test["fold"] = fold_id
            scored_folds.append(scored_test)
        scored_all = pd.concat(scored_folds, ignore_index=True)
        metrics = evaluate_predictions_with_threshold(scored_all, "score", target_col)
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
