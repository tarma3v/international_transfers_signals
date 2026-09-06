"""T25: anchor-preserving probability maps for the T24 compact h20 rank."""
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
from research.temperature_t24_history_h20_anchor import (
    CAL_SPLIT,
    COMPACT_FEATURES,
    EVAL_ORIGIN,
    FIT_ORIGIN,
    _maturity,
    _publication_lookup,
)


OUT = Path("results/research/temperature/t25_anchor_preserving_map")
REGISTERED = Path("research/temperature_t25_anchor_preserving_map_registered.md")
ALPHAS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.40)
ALPHA_NAMES = {
    0.01: "residual_a001",
    0.025: "residual_a0025",
    0.05: "residual_a005",
    0.10: "residual_a010",
    0.20: "residual_a020",
    0.40: "residual_a040",
}


def _alpha_name(alpha):
    return ALPHA_NAMES[alpha]


RESIDUAL_MODELS = tuple(_alpha_name(alpha) for alpha in ALPHAS)
CANDIDATES = (
    *RESIDUAL_MODELS,
    "daily_permute_blend50", "daily_permute",
    "copula_global_blend50", "copula_global", "copula_currency",
)
MODELS = ("identity_early", "anchor_map_selected", *CANDIDATES)
PRIORITY = {name: i for i, name in enumerate(CANDIDATES)}


def _fit_compact(X, names, index, target, maturity):
    dates = np.asarray([row[2] for row in index], dtype=object)
    cutoff = (FIT_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    train = (dates < FIT_ORIGIN) & np.isfinite(target) & (maturity < cutoff)
    columns = [names.index(name) for name in COMPACT_FEATURES]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1, penalty="l2", solver="lbfgs", max_iter=5000,
            random_state=20260906),
    )
    model.fit(X[train][:, columns], target[train].astype(int))
    raw = model.predict_proba(X[:, columns])[:, 1]
    return raw, {
        "training_rows": int(train.sum()), "fit_cutoff_ord": cutoff,
        "features": list(COMPACT_FEATURES),
        "intercept": float(model[-1].intercept_[0]),
        "coefficients": [float(value) for value in model[-1].coef_[0]],
    }


def _empirical_map(reference_score, reference_probability, score):
    order = np.argsort(reference_score, kind="mergesort")
    x = np.asarray(reference_score, dtype=float)[order]
    y = np.sort(np.asarray(reference_probability, dtype=float))
    # Collapse repeated x values so interpolation remains deterministic.
    unique, first, counts = np.unique(x, return_index=True, return_counts=True)
    mapped = np.asarray([
        float(y[start:start + count].mean())
        for start, count in zip(first, counts)
    ])
    return _clip(np.interp(score, unique, mapped, left=mapped[0], right=mapped[-1]))


def _daily_permute(frame, score):
    output = np.empty(len(frame), dtype=float)
    for _day, positions in frame.groupby("query_date", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        score_order = positions[np.argsort(score[positions], kind="mergesort")]
        output[score_order] = np.sort(
            frame.identity_early.to_numpy(dtype=float)[positions])
    return output


def _build_maps(query, compact_score, calibration):
    identity = query.identity_early.to_numpy(dtype=float)
    frozen_logit = _logit(identity)
    compact_logit = _logit(compact_score)
    design = np.column_stack([
        np.ones(int(calibration.sum())), frozen_logit[calibration]])
    intercept, slope = np.linalg.lstsq(
        design, compact_logit[calibration], rcond=None)[0]
    residual_cal = compact_logit[calibration] - (
        intercept + slope * frozen_logit[calibration])
    residual_mean = float(residual_cal.mean())
    residual_scale = float(residual_cal.std(ddof=0))
    if not np.isfinite(residual_scale) or residual_scale < 1e-8:
        residual_scale = 1.0
    residual_z = (
        compact_logit - (intercept + slope * frozen_logit) - residual_mean
    ) / residual_scale
    output = {
        _alpha_name(alpha): _clip(expit(frozen_logit + alpha * residual_z))
        for alpha in ALPHAS
    }

    output["copula_global"] = _empirical_map(
        compact_score[calibration], identity[calibration], compact_score)
    currency_map = np.empty(len(query), dtype=float)
    for currency in CORRIDORS:
        rows = query.currency.eq(currency).to_numpy()
        reference = rows & calibration
        currency_map[rows] = _empirical_map(
            compact_score[reference], identity[reference], compact_score[rows])
    output["copula_currency"] = currency_map
    output["daily_permute"] = _daily_permute(query, compact_score)
    output["daily_permute_blend50"] = _clip(expit(
        .5 * frozen_logit + .5 * _logit(output["daily_permute"])))
    output["copula_global_blend50"] = _clip(expit(
        .5 * frozen_logit + .5 * _logit(output["copula_global"])))
    details = {
        "residual_intercept": float(intercept),
        "residual_slope": float(slope),
        "residual_mean": residual_mean,
        "residual_scale": residual_scale,
    }
    return output, details


def _select(query, maps, selection):
    identity_auc = _fast_auc(
        query.loc[selection, "target"], query.loc[selection, "identity_early"])
    rows = []
    for name in CANDIDATES:
        metrics = _probability_metrics(
            query.loc[selection, "target"], maps[name][selection],
            query.loc[selection, "identity_early"])
        metrics["ece_delta"] = metrics["ece"] - metrics["baseline_ece"]
        metrics["model"] = name
        metrics["feasible"] = bool(
            metrics["auc"] > identity_auc
            and metrics["brier_delta"] <= 0.0
            and metrics["logloss_delta"] <= 0.0
            and metrics["ece_delta"] <= 0.01)
        rows.append(metrics)
    feasible = [row for row in rows if row["feasible"]]
    selected = (
        sorted(feasible, key=lambda row: (-row["auc"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "identity_early"
    )
    return selected, rows


def _paired_intervals(frame):
    ordered = frame.sort_values(["query_date", "currency"])
    dates = ordered.query_date.drop_duplicates().to_numpy()
    if len(ordered) != len(dates) * len(CORRIDORS):
        raise AssertionError("unexpected rows per date")
    y = ordered.target.to_numpy(dtype=int).reshape(len(dates), len(CORRIDORS))
    base = ordered.identity_early.to_numpy().reshape(len(dates), len(CORRIDORS))
    primary = ordered.anchor_map_selected.to_numpy().reshape(
        len(dates), len(CORRIDORS))
    rows = []
    for block in BLOCKS:
        rng = np.random.default_rng(20260910 + block)
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
    X, names, index, series, _trajectory, _trajectory_names, _paths = (
        load_round5_features())
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index)
    raw_publication, model_details = _fit_compact(
        X, names, index, target, maturity)

    snapshots = _load_snapshots()
    selected_queries = _selected_queries(
        _query_grid(snapshots), snapshots, "no_same_day_receipt")
    query = _feature_frame(selected_queries, series, _target_arrays(series))
    query = query[query.clock.eq("00:15") & np.isfinite(query.target)].copy()
    query["identity_early"] = query.identity_h20
    publication_rows, publication_dates = _publication_lookup(index, query)
    query["publication_date"] = publication_dates
    compact_score = raw_publication[publication_rows]

    date = pd.to_datetime(query.query_date).dt.date.to_numpy()
    cal_cutoff = (CAL_SPLIT - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    eval_cutoff = (EVAL_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    calibration = (
        (date >= FIT_ORIGIN) & (date < CAL_SPLIT)
        & (query.maturity_ord.to_numpy() < cal_cutoff))
    selection = (
        (date >= CAL_SPLIT) & (date < EVAL_ORIGIN)
        & (query.maturity_ord.to_numpy() < eval_cutoff))
    evaluation = date >= EVAL_ORIGIN
    maps, map_details = _build_maps(query, compact_score, calibration)
    selected_model, screen = _select(query, maps, selection)
    for name, values in maps.items():
        query[name] = values
    query["anchor_map_selected"] = (
        query.identity_early if selected_model == "identity_early"
        else query[selected_model])
    evaluated = query[evaluation].copy()

    metrics = []
    for name in MODELS:
        for slice_name, group, part in _slices(evaluated):
            result = _probability_metrics(
                part.target, part[name], part.identity_early)
            result["ece_delta"] = result["ece"] - result["baseline_ece"]
            metrics.append({
                "model": name, "slice": slice_name, "group": group, **result})
    metrics = pd.DataFrame(metrics)
    intervals = _paired_intervals(evaluated)
    primary = metrics[
        metrics.model.eq("anchor_map_selected") & metrics.slice.eq("ALL")
    ].iloc[0]
    identity_auc = _fast_auc(evaluated.target, evaluated.identity_early)
    auc_bounds = intervals[intervals.metric.eq("auc")]
    brier_bounds = intervals[intervals.metric.eq("brier")]
    passed = bool(
        primary.auc > identity_auc and primary.brier_delta < 0.0
        and primary.logloss_delta < 0.0 and primary.ece_delta <= .01
        and (auc_bounds.ci_low > 0.0).all()
        and (brier_bounds.ci_high < 0.0).all())
    local = metrics[
        metrics.slice.eq("currency_year")
        & metrics.model.isin(["identity_early", "anchor_map_selected"])].copy()
    local["high_ece"] = local.ece > .08
    high_ece = local.groupby("model").high_ece.sum().to_dict()
    permute_error = evaluated.groupby("query_date").apply(
        lambda part: float(np.max(np.abs(
            np.sort(part.daily_permute.to_numpy())
            - np.sort(part.identity_early.to_numpy())))))
    source_files = [
        REGISTERED,
        Path("research/temperature_t25_anchor_preserving_map.py"),
        Path("research/temperature_t24_history_h20_anchor.py"),
        Path("research/cache/extended_features_2010_2026.npz"),
        Path("data/cbr_rates_2010_2026.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
    ]
    metadata = {
        "packet": "temperature-T25",
        "selected_model": selected_model,
        "selection_on_open_period": False,
        "passed": passed,
        "evaluation_rows": int(len(evaluated)),
        "evaluation_dates": int(evaluated.query_date.nunique()),
        "screen": screen,
        "map_details": map_details,
        "daily_permute_max_multiset_error": float(permute_error.max()),
        "high_ece_currency_year_rows": {
            key: int(value) for key, value in high_ece.items()},
        "checks": {
            "all_training_labels_mature": True,
            "mapping_reference_labels_mature": bool(
                query.loc[calibration, "maturity_ord"].max() < cal_cutoff),
            "selection_labels_mature": bool(
                query.loc[selection, "maturity_ord"].max() < eval_cutoff),
            "publication_date_no_later_than_query": bool((
                query.publication_date <= pd.to_datetime(query.query_date).dt.date).all()),
            "all_horizon_sources_no_later_than_query": bool((
                pd.to_datetime(query.source_at, utc=True)
                <= pd.to_datetime(query.query_at, utc=True)).all()),
            "daily_permute_preserves_multiset": bool(permute_error.max() < 1e-12),
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
    model_details["map_details"] = map_details
    return {
        "predictions": evaluated[[
            "query_date", "year", "currency", "query_at", "source_at",
            "publication_date", "target", "maturity_ord", *MODELS,
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
        "daily_permute_max_multiset_error": result["metadata"][
            "daily_permute_max_multiset_error"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
