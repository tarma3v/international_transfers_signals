"""T27: extend delayed h20 calibration with causal 2023 OOS history."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import build_targets
from research.round5_features import load_round5_features
from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS
from research.temperature_t21_h20_curve_head import _clip, _logit
from research.temperature_t24_history_h20_anchor import COMPACT_FEATURES, _maturity
from research.temperature_t26_delayed_base_rate import (
    CANDIDATES,
    MODELS,
    _build_query,
    _delayed_candidates,
    _metrics,
    _paired_intervals,
    _selection_metrics,
)


OUT = Path("results/research/temperature/t27_rolling_origin_history")
REGISTERED = Path("research/temperature_t27_rolling_origin_history_registered.md")
T4_OUT = Path("results/research/temperature/t4_premarket/outputs.npz")
T4_META = Path("results/research/temperature/t4_premarket/metadata.json")
HISTORY_START = dt.date(2023, 1, 1)
HISTORY_END = dt.date(2024, 1, 1)
ALPHA = 0.40


def _quarter_end(origin):
    if origin.month == 10:
        return dt.date(origin.year + 1, 1, 1)
    return dt.date(origin.year, origin.month + 3, 1)


def _rolling_compact_scores(X, names, index, target, maturity):
    dates = np.asarray([row[2] for row in index], dtype=object)
    columns = [names.index(name) for name in COMPACT_FEATURES]
    compact = np.asarray(X[:, columns], dtype=float)
    finite = np.all(np.isfinite(compact), axis=1)
    output = np.full(len(index), np.nan)
    logs = []
    for month in (1, 4, 7, 10):
        origin = dt.date(2023, month, 1)
        end = _quarter_end(origin)
        cutoff = (origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
        train = (
            (dates < origin)
            & np.isfinite(target)
            & np.isfinite(maturity)
            & (maturity < cutoff)
            & finite
        )
        query = (dates >= origin) & (dates < end) & finite
        if train.sum() < 1000 or np.unique(target[train]).size < 2:
            raise AssertionError(f"insufficient compact history at {origin}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                random_state=20260906,
            ),
        )
        model.fit(compact[train], target[train].astype(int))
        output[query] = model.predict_proba(compact[query])[:, 1]
        logs.append({
            "origin": origin,
            "end": end,
            "cutoff_ord": cutoff,
            "training_rows": int(train.sum()),
            "query_rows": int(query.sum()),
            "latest_training_date": max(dates[train]),
            "latest_training_maturity_ord": int(np.max(maturity[train])),
            "intercept": float(model[-1].intercept_[0]),
        })
    return output, pd.DataFrame(logs)


def _historical_feedback(t25_map_details):
    X, names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index)
    compact_score, fit_log = _rolling_compact_scores(
        X, names, index, target, maturity)

    with np.load(T4_OUT, allow_pickle=True) as archive:
        if not np.array_equal(archive["dates"], dates):
            raise AssertionError("T4 dates no longer align with feature index")
        if not np.array_equal(archive["currencies"].astype(str), currencies.astype(str)):
            raise AssertionError("T4 currencies no longer align with feature index")
        identity = archive["prob__history_hist__h20"].astype(float)
        t4_train_count = archive["model_n_train_20"].astype(int)

    history = (dates >= HISTORY_START) & (dates < HISTORY_END)
    valid = (
        history
        & np.isfinite(identity)
        & np.isfinite(compact_score)
        & np.isfinite(target)
        & np.isfinite(maturity)
    )
    if valid.sum() != history.sum():
        raise AssertionError("2023 rolling history is incomplete")
    details = t25_map_details
    residual_z = (
        _logit(compact_score[valid])
        - (details["residual_intercept"] + details["residual_slope"] * _logit(identity[valid]))
        - details["residual_mean"]
    ) / details["residual_scale"]
    t25_base = _clip(expit(_logit(identity[valid]) + ALPHA * residual_z))
    frame = pd.DataFrame({
        "query_date": dates[valid],
        "publication_date": dates[valid],
        "currency": currencies[valid],
        "target": target[valid].astype(int),
        "maturity_ord": maturity[valid].astype(int),
        "identity_early": identity[valid],
        "t25_base": t25_base,
        "year": 2023,
        "t4_model_n_train": t4_train_count[valid],
        "is_historical_feedback": True,
    })
    diagnostics = {
        "rows": int(len(frame)),
        "dates": int(frame.query_date.nunique()),
        "first_date": str(frame.query_date.min()),
        "last_date": str(frame.query_date.max()),
        "minimum_t4_training_rows": int(frame.t4_model_n_train.min()),
        "keys_unique": bool(not frame[["publication_date", "currency"]].duplicated().any()),
        "probabilities_finite": bool(np.isfinite(frame[["identity_early", "t25_base"]]).all().all()),
    }
    return frame, fit_log, diagnostics


def run():
    query, query_selection, query_evaluation, model_details, split = _build_query()
    history, fit_log, history_details = _historical_feedback(
        model_details["t25_map_details"])
    query = query.copy()
    query["is_historical_feedback"] = False
    combined = pd.concat([
        history,
        query[[
            "query_date", "publication_date", "currency", "target",
            "maturity_ord", "identity_early", "t25_base", "year",
            "is_historical_feedback",
        ]],
    ], ignore_index=True, sort=False)

    n_history = len(history)
    selection = np.r_[np.zeros(n_history, dtype=bool), query_selection]
    evaluation = np.r_[np.zeros(n_history, dtype=bool), query_evaluation]
    candidates, states, feedback_details = _delayed_candidates(
        combined, combined.t25_base)
    selected_model, screen = _selection_metrics(combined, candidates, selection)
    for name, values in candidates.items():
        query[name] = values[n_history:]
    query["delayed_selected"] = (
        query.t25_base if selected_model == "t25_base" else query[selected_model]
    )
    evaluated = query[query_evaluation].copy()
    metrics = _metrics(evaluated)
    intervals = pd.concat([
        _paired_intervals(evaluated, "delayed_selected", "t25_base", "t25"),
        _paired_intervals(evaluated, "delayed_selected", "identity_early", "identity"),
    ], ignore_index=True)

    primary = metrics[(metrics.model == "delayed_selected") & (metrics.slice == "ALL")].iloc[0]
    year_rows = metrics[(metrics.model == "delayed_selected") & (metrics.slice == "year")]
    currency_rows = metrics[(metrics.model == "delayed_selected") & (metrics.slice == "currency")]
    brier_intervals = intervals[(intervals.comparison == "t25") & (intervals.metric == "brier")]
    passed = bool(
        primary.brier_delta < 0.0
        and primary.logloss_delta < 0.0
        and primary.ece_delta_t25 <= 0.005
        and primary.auc_delta_t25 >= -0.005
        and (brier_intervals.ci_high < 0.0).all()
        and (year_rows.brier_delta <= 0.0).all()
        and (currency_rows.brier_delta <= 0.005).all()
    )

    fit_causal = bool((
        fit_log.latest_training_maturity_ord < fit_log.cutoff_ord
    ).all() and (
        pd.to_datetime(fit_log.latest_training_date).dt.date
        < pd.to_datetime(fit_log.origin).dt.date
    ).all())
    used_states = states[states.latest_feedback_maturity_ord.notna()]
    feedback_causal = bool((
        used_states.latest_feedback_maturity_ord < used_states.cutoff_ord
    ).all())
    source_files = [
        REGISTERED,
        Path("research/temperature_t27_rolling_origin_history.py"),
        Path("research/temperature_t26_delayed_base_rate.py"),
        Path("research/temperature_t25_anchor_preserving_map.py"),
        Path("research/temperature_t24_history_h20_anchor.py"),
        T4_OUT,
        T4_META,
        Path("research/cache/extended_features_2010_2026.npz"),
        Path("data/cbr_rates_2010_2026.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
    ]
    metadata = {
        "packet": "temperature-T27",
        "selected_model": selected_model,
        "t25_base_model": model_details["t25_selected"],
        "selection_on_open_period": False,
        "open_period_previously_inspected": True,
        "fresh_independent_holdout": False,
        "passed": passed,
        "evaluation_rows": int(len(evaluated)),
        "evaluation_dates": int(evaluated.query_date.nunique()),
        "screen": screen,
        "split": split,
        "history_details": history_details,
        "feedback_details": feedback_details,
        "checks": {
            "t25_selection_rebuilt": True,
            "t4_alignment_exact": True,
            "historical_compact_fit_causal": fit_causal,
            "historical_feedback_unique": history_details["keys_unique"],
            "feedback_trace_causal": feedback_causal,
            "publication_date_no_later_than_query": bool((
                pd.to_datetime(query.publication_date)
                <= pd.to_datetime(query.query_date)).all()),
            "all_horizon_sources_no_later_than_query": bool((
                pd.to_datetime(query.source_at, utc=True)
                <= pd.to_datetime(query.query_at, utc=True)).all()),
            "opened_evaluation_only": True,
            "changes_push_policy": False,
            "historical_receipts_certified": False,
            "bank_execution_validated": False,
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    model_details.update({
        "historical_compact_fit_log": fit_log.assign(
            origin=fit_log.origin.astype(str),
            end=fit_log.end.astype(str),
            latest_training_date=fit_log.latest_training_date.astype(str),
        ).to_dict("records"),
        "historical_alpha": ALPHA,
    })
    return {
        "predictions": evaluated[[
            "query_date", "year", "currency", "query_at", "source_at",
            "publication_date", "target", "maturity_ord", *MODELS,
        ]],
        "historical_feedback": history,
        "historical_fit_log": fit_log,
        "states": states,
        "metrics": metrics,
        "paired_bootstrap": intervals,
        "model_details": model_details,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["historical_feedback"].to_csv(
        OUT / "historical_feedback.csv.gz", index=False, compression="gzip")
    result["historical_fit_log"].to_csv(OUT / "historical_fit_log.csv", index=False)
    result["states"].to_csv(OUT / "states.csv.gz", index=False, compression="gzip")
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    (OUT / "model_details.json").write_text(json.dumps(
        result["model_details"], ensure_ascii=False, indent=2))
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    overall = result["metrics"][result["metrics"].slice == "ALL"]
    print(json.dumps({
        "selected_model": result["metadata"]["selected_model"],
        "passed": result["metadata"]["passed"],
        "history_details": result["metadata"]["history_details"],
        "screen": result["metadata"]["screen"],
        "evaluation": overall.set_index("model")[[
            "auc", "average_precision", "brier", "logloss", "ece",
            "brier_delta", "logloss_delta", "ece_delta_t25", "auc_delta_t25",
        ]].to_dict("index"),
        "paired_bootstrap": result["paired_bootstrap"].to_dict("records"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
