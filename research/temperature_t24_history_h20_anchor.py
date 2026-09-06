"""T24: a history-only multiscale h20 anchor for overnight/premarket use."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from ml.targets import build_targets
from research.round5_features import load_round5_features
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    BOOTSTRAP_DRAWS,
    EMBARGO_DAYS,
    _load_snapshots,
    _probability_metrics,
    _query_grid,
    _selected_queries,
    _target_arrays,
)
from research.temperature_t21_h20_curve_head import (
    _clip,
    _fast_auc,
    _feature_frame,
    _logit,
)


OUT = Path("results/research/temperature/t24_history_h20_anchor")
REGISTERED = Path("research/temperature_t24_history_h20_anchor_registered.md")
FIT_ORIGIN = dt.date(2024, 1, 1)
CAL_SPLIT = dt.date(2024, 7, 1)
EVAL_ORIGIN = dt.date(2025, 1, 1)
MODELS = (
    "identity_early", "compact_logit", "extended_hgb", "recent_hgb",
    "path_extra",
)
COMPLEXITY = {
    "compact_logit": 0, "extended_hgb": 1, "recent_hgb": 2, "path_extra": 3,
}
COMPACT_FEATURES = (
    "ret_5", "ret_10", "ret_20", "ret_60", "ret_120", "ret_250",
    "pct_range_30", "pct_range_90", "pct_range_180",
    "days_beaten_30", "days_beaten_90", "days_beaten_180",
    "z_30", "z_90", "z_180", "vol_10", "vol_30", "vol_90", "vol_ratio",
    "vs_sma_20", "vs_sma_60", "accel_5_20", "accel_20_60",
    "peer_ret_5_mean", "rel_to_peers_5", "peer_dispersion_5",
    "usd_ret_5", "usd_ret_20", "cny_ret_5", "cny_ret_20",
    "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    "gap_days", "is_after_gap", "currency_TJS", "currency_UZS",
    "currency_KGS", "currency_AMD", "currency_KZT",
)


def _maturity(series, index, h=20):
    output = np.full(len(index), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        if position + h < len(series[currency]):
            output[row] = series[currency].dates[position + h].toordinal()
    return output


def _fit_models(X, trajectory, names, index, y, maturity):
    dates = np.asarray([row[2] for row in index], dtype=object)
    cutoff = (FIT_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    train = (dates < FIT_ORIGIN) & np.isfinite(y) & (maturity < cutoff)
    compact_indices = [names.index(name) for name in COMPACT_FEATURES]
    y_train = y[train].astype(int)
    details = {"training_rows": int(train.sum()), "fit_cutoff_ord": cutoff}

    compact = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1, penalty="l2", solver="lbfgs", max_iter=5000,
            random_state=20260906),
    )
    compact.fit(X[train][:, compact_indices], y_train)

    extended = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=200, max_depth=2,
        min_samples_leaf=80, l2_regularization=20.0,
        random_state=20260906)
    extended.fit(X[train], y_train)

    recent = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=200, max_depth=2,
        min_samples_leaf=80, l2_regularization=20.0,
        random_state=20260906)
    age_days = np.asarray([(FIT_ORIGIN - day).days for day in dates[train]])
    sample_weight = np.maximum(0.15, np.exp(-np.log(2.0) * age_days / 730.0))
    recent.fit(X[train], y_train, sample_weight=sample_weight)

    path = ExtraTreesClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=40,
        max_features=0.5, n_jobs=1, random_state=20260906)
    path.fit(trajectory[train], y_train)

    raw = {
        "compact_logit": compact.predict_proba(X[:, compact_indices])[:, 1],
        "extended_hgb": extended.predict_proba(X)[:, 1],
        "recent_hgb": recent.predict_proba(X)[:, 1],
        "path_extra": path.predict_proba(trajectory)[:, 1],
    }
    details.update({
        "compact_features": list(COMPACT_FEATURES),
        "all_feature_count": int(X.shape[1]),
        "path_feature_count": int(trajectory.shape[1]),
        "recent_weight_min": float(sample_weight.min()),
        "recent_weight_max": float(sample_weight.max()),
        "compact_intercept": float(compact[-1].intercept_[0]),
        "compact_coefficients": [float(v) for v in compact[-1].coef_[0]],
        "path_feature_importances": [float(v) for v in path.feature_importances_],
    })
    return raw, details


def _publication_lookup(index, query):
    by_currency = {}
    for currency in CORRIDORS:
        positions = [i for i, row in enumerate(index) if row[0] == currency]
        dates = np.asarray([index[i][2] for i in positions], dtype=object)
        by_currency[currency] = (dates, np.asarray(positions, dtype=int))
    rows = np.empty(len(query), dtype=int)
    publication_dates = np.empty(len(query), dtype=object)
    for currency in CORRIDORS:
        mask = query.currency.eq(currency).to_numpy()
        dates, positions = by_currency[currency]
        query_dates = np.asarray(
            [pd.Timestamp(value).date() for value in query.loc[mask, "query_date"]],
            dtype=object)
        located = np.searchsorted(dates, query_dates, side="right") - 1
        if (located < 0).any():
            raise AssertionError("query predates feature history")
        rows[mask] = positions[located]
        publication_dates[mask] = dates[located]
    return rows, publication_dates


def _platt(raw_calibration, y_calibration, raw_all):
    model = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=3000,
        random_state=20260906)
    model.fit(_logit(raw_calibration)[:, None], y_calibration)
    return _clip(model.predict_proba(_logit(raw_all)[:, None])[:, 1]), {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def _calibrate_and_select(query, raw_query):
    date = pd.to_datetime(query.query_date).dt.date.to_numpy()
    cal_cutoff = (CAL_SPLIT - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    eval_cutoff = (EVAL_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    calibration = (
        (date >= FIT_ORIGIN) & (date < CAL_SPLIT)
        & (query.maturity_ord.to_numpy() < cal_cutoff)
    )
    selection = (
        (date >= CAL_SPLIT) & (date < EVAL_ORIGIN)
        & (query.maturity_ord.to_numpy() < eval_cutoff)
    )
    if calibration.sum() < 200 or selection.sum() < 200:
        raise AssertionError("insufficient calibration/selection rows")
    calibrated = {}
    mappings = {}
    screen_rows = []
    y_cal = query.loc[calibration, "target"].to_numpy(dtype=int)
    for name, values in raw_query.items():
        calibrated[name], mappings[name] = _platt(
            values[calibration], y_cal, values)
        metrics = _probability_metrics(
            query.loc[selection, "target"], calibrated[name][selection],
            query.loc[selection, "identity_early"])
        metrics["ece_delta"] = metrics["ece"] - metrics["baseline_ece"]
        metrics["model"] = name
        metrics["feasible"] = bool(
            metrics["auc"] > _fast_auc(
                query.loc[selection, "target"],
                query.loc[selection, "identity_early"])
            and metrics["brier_delta"] <= 0.0
            and metrics["logloss_delta"] <= 0.0
            and metrics["ece_delta"] <= 0.01
        )
        screen_rows.append(metrics)
    feasible = [row for row in screen_rows if row["feasible"]]
    selected = (
        sorted(feasible, key=lambda row: (-row["auc"], COMPLEXITY[row["model"]]))[0]["model"]
        if feasible else "identity_early"
    )
    return calibrated, selected, mappings, screen_rows, {
        "calibration_rows": int(calibration.sum()),
        "selection_rows": int(selection.sum()),
        "calibration_cutoff_ord": cal_cutoff,
        "selection_cutoff_ord": eval_cutoff,
        "latest_calibration_maturity_ord": int(
            query.loc[calibration, "maturity_ord"].max()),
        "latest_selection_maturity_ord": int(
            query.loc[selection, "maturity_ord"].max()),
    }


def _paired_intervals(frame, model):
    ordered = frame.sort_values(["query_date", "currency"])
    dates = ordered.query_date.drop_duplicates().to_numpy()
    if len(ordered) != len(dates) * len(CORRIDORS):
        raise AssertionError("unexpected rows per date")
    y = ordered.target.to_numpy(dtype=int).reshape(len(dates), len(CORRIDORS))
    base = ordered.identity_early.to_numpy().reshape(len(dates), len(CORRIDORS))
    primary = ordered[model].to_numpy().reshape(len(dates), len(CORRIDORS))
    rows = []
    for block in BLOCKS:
        rng = np.random.default_rng(20260909 + block)
        blocks = int(np.ceil(len(dates) / block))
        starts = rng.integers(0, len(dates), size=(BOOTSTRAP_DRAWS, blocks))
        offsets = np.arange(block)
        samples = (
            (starts[:, :, None] + offsets[None, None, :]) % len(dates)
        ).reshape(BOOTSTRAP_DRAWS, -1)[:, :len(dates)]
        brier_delta = np.empty(BOOTSTRAP_DRAWS)
        auc_delta = np.empty(BOOTSTRAP_DRAWS)
        for draw_i, sample in enumerate(samples):
            ys = y[sample].ravel()
            bs = base[sample].ravel()
            ps = primary[sample].ravel()
            brier_delta[draw_i] = np.mean((ps - ys) ** 2 - (bs - ys) ** 2)
            auc_delta[draw_i] = _fast_auc(ys, ps) - _fast_auc(ys, bs)
        for metric, values in (("brier", brier_delta), ("auc", auc_delta)):
            rows.append({
                "metric": metric, "block_dates": block,
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, .025)),
                "ci_high": float(np.quantile(values, .975)),
            })
    return pd.DataFrame(rows)


def _slices(frame):
    yield "ALL", "ALL", frame
    for currency, part in frame.groupby("currency", sort=True):
        yield "currency", currency, part
    for year, part in frame.groupby("year", sort=True):
        yield "year", str(year), part
    for (currency, year), part in frame.groupby(["currency", "year"], sort=True):
        yield "currency_year", f"{currency}:{year}", part


def run():
    X, names, index, series, trajectory, trajectory_names, _paths = (
        load_round5_features())
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index)
    raw_publication, model_details = _fit_models(
        X, trajectory, names, index, target, maturity)

    snapshots = _load_snapshots()
    selected = _selected_queries(
        _query_grid(snapshots), snapshots, "no_same_day_receipt")
    query = _feature_frame(selected, series, _target_arrays(series))
    query = query[query.clock.eq("00:15") & np.isfinite(query.target)].copy()
    query["identity_early"] = query.identity_h20
    publication_rows, publication_dates = _publication_lookup(index, query)
    query["publication_date"] = publication_dates
    if not (query.publication_date <= pd.to_datetime(query.query_date).dt.date).all():
        raise AssertionError("future publication mapped to query")
    raw_query = {
        name: values[publication_rows] for name, values in raw_publication.items()}
    calibrated, selected_model, mappings, screen, split_checks = (
        _calibrate_and_select(query, raw_query))
    for name, values in calibrated.items():
        query[name] = values
    query["history_h20_selected"] = (
        query.identity_early if selected_model == "identity_early"
        else query[selected_model])
    evaluation = query[
        pd.to_datetime(query.query_date).dt.date >= EVAL_ORIGIN].copy()

    metrics = []
    evaluation_models = ["identity_early", "history_h20_selected", *MODELS[1:]]
    for name in evaluation_models:
        for slice_name, group, part in _slices(evaluation):
            result = _probability_metrics(
                part.target, part[name], part.identity_early)
            result["ece_delta"] = result["ece"] - result["baseline_ece"]
            metrics.append({
                "model": name, "slice": slice_name, "group": group, **result})
    metrics = pd.DataFrame(metrics)
    intervals = _paired_intervals(evaluation, "history_h20_selected")
    primary_metrics = metrics[
        metrics.model.eq("history_h20_selected") & metrics.slice.eq("ALL")
    ].iloc[0]
    auc_bounds = intervals[intervals.metric.eq("auc")]
    brier_bounds = intervals[intervals.metric.eq("brier")]
    # _probability_metrics does not expose baseline AUC; compare explicitly.
    identity_auc = _fast_auc(evaluation.target, evaluation.identity_early)
    passed = bool(
        primary_metrics.auc > identity_auc
        and primary_metrics.brier_delta < 0.0
        and primary_metrics.logloss_delta < 0.0
        and primary_metrics.ece_delta <= .01
        and (auc_bounds.ci_low > 0.0).all()
        and (brier_bounds.ci_high < 0.0).all()
    )
    local = metrics[
        metrics.slice.eq("currency_year")
        & metrics.model.isin(["identity_early", "history_h20_selected"])].copy()
    local["high_ece"] = local.ece > .08
    high_ece = local.groupby("model").high_ece.sum().to_dict()
    source_files = [
        REGISTERED,
        Path("research/temperature_t24_history_h20_anchor.py"),
        Path("research/round5_features.py"),
        Path("research/extended_features.py"),
        Path("research/cache/extended_features_2010_2026.npz"),
        Path("research/cache/round5_path_features.npz"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
        Path("data/cbr_rates_2010_2026.json"),
    ]
    metadata = {
        "packet": "temperature-T24",
        "selected_model": selected_model,
        "selection_on_open_period": False,
        "evaluation_rows": int(len(evaluation)),
        "evaluation_dates": int(evaluation.query_date.nunique()),
        "passed": passed,
        "screen": screen,
        "split_checks": split_checks,
        "high_ece_currency_year_rows": {
            key: int(value) for key, value in high_ece.items()},
        "checks": {
            "all_training_labels_mature": True,
            "calibration_and_selection_labels_mature": bool(
                split_checks["latest_calibration_maturity_ord"]
                < split_checks["calibration_cutoff_ord"]
                and split_checks["latest_selection_maturity_ord"]
                < split_checks["selection_cutoff_ord"]),
            "splits_disjoint": True,
            "publication_date_no_later_than_query": True,
            "weekends_hold_last_publication": True,
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
        "trajectory_feature_names": trajectory_names,
        "platt_mappings": mappings,
    })
    return {
        "predictions": evaluation[[
            "query_date", "year", "currency", "query_at", "source_at",
            "publication_date", "target", "maturity_ord", "identity_early",
            "history_h20_selected", *MODELS[1:],
        ]],
        "metrics": metrics, "paired_bootstrap": intervals,
        "model_details": model_details, "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(
        OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(
        OUT / "paired_bootstrap.csv", index=False)
    (OUT / "model_details.json").write_text(json.dumps(
        result["model_details"], ensure_ascii=False, indent=2))
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    overall = result["metrics"][result["metrics"].slice.eq("ALL")]
    print(json.dumps({
        "selected_model": result["metadata"]["selected_model"],
        "passed": result["metadata"]["passed"],
        "screen": result["metadata"]["screen"],
        "evaluation": overall.set_index("model")[[
            "auc", "average_precision", "brier", "logloss", "ece",
            "brier_delta", "logloss_delta", "ece_delta",
        ]].to_dict("index"),
        "paired_bootstrap": result["paired_bootstrap"].to_dict("records"),
        "high_ece_currency_year_rows": result["metadata"][
            "high_ece_currency_year_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
