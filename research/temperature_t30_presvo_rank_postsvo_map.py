"""T30: pre-2022 h20 rank with a frozen post-SVO 2022 probability map."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import build_targets
from research.round5_features import load_round5_features
from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS, _probability_metrics
from research.temperature_t21_h20_curve_head import _clip, _fast_auc, _logit
from research.temperature_t24_history_h20_anchor import COMPACT_FEATURES, _maturity
from research.temperature_t26_delayed_base_rate import _build_query, _paired_intervals


OUT = Path("results/research/temperature/t30_presvo_rank_postsvo_map")
REGISTERED = Path("research/temperature_t30_presvo_rank_postsvo_map_registered.md")
RANK_ORIGIN = dt.date(2022, 1, 1)
MAP_START = dt.date(2022, 4, 1)
MAP_END = dt.date(2022, 12, 1)
SCREEN_START = dt.date(2023, 1, 1)
VALIDATION_START = dt.date(2024, 1, 1)
EVALUATION_START = dt.date(2025, 1, 1)
FAMILIES = {
    "all": None,
    "recent4y": dt.date(2018, 1, 1),
    "recent2y": dt.date(2020, 1, 1),
}
BETAS = (0.25, 0.50, 0.75, 1.00)
PLATT_RIDGE = 5.0
PLATT_INTERCEPT_LIMIT = 4.0
PLATT_SLOPE_LIMITS = (0.05, 5.0)


def _candidate_names():
    names = []
    for family in FAMILIES:
        names.append(f"{family}_raw")
        names.extend(f"{family}_platt_b{int(beta * 100):03d}" for beta in BETAS)
    return tuple(names)


CANDIDATES = _candidate_names()
PRIORITY = {name: i for i, name in enumerate(CANDIDATES)}
MODELS = ("identity_early", "t25_base", "t30_candidate", "t30_selected", *CANDIDATES)


def _fit_positive_platt(raw_probability, target):
    x = _logit(np.asarray(raw_probability, dtype=float))
    y = np.asarray(target, dtype=float)

    def objective(theta):
        intercept = theta[0]
        slope = np.exp(theta[1])
        z = intercept + slope * x
        loss = np.logaddexp(0.0, z).sum() - np.dot(y, z)
        penalty = PLATT_RIDGE * (intercept ** 2 + (slope - 1.0) ** 2)
        return float(loss + penalty)

    bounds = [
        (-PLATT_INTERCEPT_LIMIT, PLATT_INTERCEPT_LIMIT),
        (np.log(PLATT_SLOPE_LIMITS[0]), np.log(PLATT_SLOPE_LIMITS[1])),
    ]
    fit = minimize(objective, np.array([0.0, 0.0]), method="L-BFGS-B", bounds=bounds)
    if not fit.success:
        raise RuntimeError(f"positive Platt fit failed: {fit.message}")
    return float(fit.x[0]), float(np.exp(fit.x[1])), float(fit.fun)


def _fit_rank_models(X, names, dates, target, maturity):
    columns = [names.index(name) for name in COMPACT_FEATURES]
    compact = np.asarray(X[:, columns], dtype=float)
    finite = np.all(np.isfinite(compact), axis=1)
    cutoff = (RANK_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    outputs = {}
    logs = []
    details = {}
    for family, start in FAMILIES.items():
        train = (
            finite & np.isfinite(target) & np.isfinite(maturity)
            & (dates < RANK_ORIGIN) & (maturity < cutoff)
        )
        if start is not None:
            train &= dates >= start
        if train.sum() < 1000 or np.unique(target[train]).size < 2:
            raise AssertionError(f"insufficient rank history for {family}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, penalty="l2", solver="lbfgs", max_iter=5000,
                random_state=20260906,
            ),
        )
        model.fit(compact[train], target[train].astype(int))
        raw = np.full(len(dates), np.nan)
        raw[finite] = model.predict_proba(compact[finite])[:, 1]
        outputs[family] = raw
        logs.append({
            "family": family,
            "training_start": min(dates[train]),
            "training_end_exclusive": RANK_ORIGIN,
            "cutoff_ord": cutoff,
            "training_rows": int(train.sum()),
            "latest_training_date": max(dates[train]),
            "latest_training_maturity_ord": int(np.max(maturity[train])),
        })
        details[family] = {
            "features": list(COMPACT_FEATURES),
            "intercept": float(model[-1].intercept_[0]),
            "coefficients": [float(value) for value in model[-1].coef_[0]],
        }
    return outputs, pd.DataFrame(logs), details


def _build_candidates(raw_models, dates, target, maturity):
    cutoff = (SCREEN_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    calibration = (
        (dates >= MAP_START) & (dates < MAP_END)
        & np.isfinite(target) & np.isfinite(maturity) & (maturity < cutoff)
    )
    candidates = {}
    map_rows = []
    for family, raw in raw_models.items():
        usable = calibration & np.isfinite(raw)
        intercept, slope, objective = _fit_positive_platt(raw[usable], target[usable])
        platt = _clip(expit(intercept + slope * _logit(raw)))
        candidates[f"{family}_raw"] = _clip(raw)
        for beta in BETAS:
            name = f"{family}_platt_b{int(beta * 100):03d}"
            candidates[name] = _clip(expit(
                (1.0 - beta) * _logit(raw) + beta * _logit(platt)))
        map_rows.append({
            "family": family,
            "map_start": MAP_START,
            "map_end_exclusive": MAP_END,
            "cutoff_ord": cutoff,
            "mapping_rows": int(usable.sum()),
            "latest_mapping_date": max(dates[usable]),
            "latest_mapping_maturity_ord": int(np.max(maturity[usable])),
            "intercept": intercept,
            "slope": slope,
            "objective": objective,
        })
    return candidates, pd.DataFrame(map_rows), calibration


def _candidate_metrics(frame, mask):
    base_auc = _fast_auc(frame.loc[mask, "target"], frame.loc[mask, "identity_early"])
    rows = []
    for name in CANDIDATES:
        result = _probability_metrics(
            frame.loc[mask, "target"], frame.loc[mask, name],
            frame.loc[mask, "identity_early"])
        result["model"] = name
        result["auc_delta"] = result["auc"] - base_auc
        result["ece_delta"] = result["ece"] - result["baseline_ece"]
        result["feasible"] = bool(
            result["auc_delta"] >= 0.02
            and result["brier_delta"] < 0.0
            and result["logloss_delta"] < 0.0
            and result["ece_delta"] <= 0.005)
        rows.append(result)
    return rows


def _choose(rows):
    feasible = [row for row in rows if row["feasible"]]
    return (sorted(feasible, key=lambda row: (
        row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "identity_early")


def _slices(frame):
    yield "ALL", "ALL", frame
    for currency, part in frame.groupby("currency", sort=True):
        yield "currency", currency, part
    for year, part in frame.groupby("year", sort=True):
        yield "year", str(year), part
    for (currency, year), part in frame.groupby(["currency", "year"], sort=True):
        yield "currency_year", f"{currency}:{year}", part


def _metrics(frame):
    rows = []
    identity_auc_cache = {}
    t25_auc_cache = {}
    for name in MODELS:
        for slice_name, group, part in _slices(frame):
            key = (slice_name, group)
            if key not in identity_auc_cache:
                identity_auc_cache[key] = _fast_auc(part.target, part.identity_early)
                t25_auc_cache[key] = _fast_auc(part.target, part.t25_base)
            identity = _probability_metrics(part.target, part[name], part.identity_early)
            t25 = _probability_metrics(part.target, part[name], part.t25_base)
            rows.append({
                "model": name, "slice": slice_name, "group": group,
                **identity,
                "auc_delta_identity": identity["auc"] - identity_auc_cache[key],
                "ece_delta_identity": identity["ece"] - identity["baseline_ece"],
                "brier_delta_t25": t25["brier_delta"],
                "logloss_delta_t25": t25["logloss_delta"],
                "auc_delta_t25": t25["auc"] - t25_auc_cache[key],
                "ece_delta_t25": t25["ece"] - t25["baseline_ece"],
            })
    return pd.DataFrame(rows)


def run():
    X, names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index)
    raw_models, fit_log, model_details = _fit_rank_models(
        X, names, dates, target, maturity)
    candidates, map_log, calibration = _build_candidates(
        raw_models, dates, target, maturity)

    with np.load("results/research/temperature/t4_premarket/outputs.npz", allow_pickle=True) as archive:
        if not np.array_equal(archive["dates"], dates):
            raise AssertionError("T4 dates no longer align")
        if not np.array_equal(archive["currencies"].astype(str), currencies.astype(str)):
            raise AssertionError("T4 currencies no longer align")
        identity = archive["prob__history_hist__h20"].astype(float)

    frame = pd.DataFrame({
        "query_date": dates,
        "publication_date": dates,
        "year": [day.year for day in dates],
        "currency": currencies,
        "target": target,
        "maturity_ord": maturity,
        "identity_early": identity,
    })
    for name, values in candidates.items():
        frame[name] = values

    valid = (
        (dates >= SCREEN_START) & np.isfinite(target) & np.isfinite(maturity)
        & np.isfinite(identity)
        & np.all(np.isfinite(frame[list(CANDIDATES)].to_numpy()), axis=1)
    )
    frame = frame.loc[valid].reset_index(drop=True)
    frame_dates = pd.to_datetime(frame.query_date).dt.date.to_numpy()
    screen_cutoff = (VALIDATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    validation_cutoff = (EVALUATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    screen = (
        (frame_dates >= SCREEN_START) & (frame_dates < VALIDATION_START)
        & (frame.maturity_ord.to_numpy() < screen_cutoff))
    validation = (
        (frame_dates >= VALIDATION_START) & (frame_dates < EVALUATION_START)
        & (frame.maturity_ord.to_numpy() < validation_cutoff))
    evaluation = frame_dates >= EVALUATION_START
    if min(screen.sum(), validation.sum(), evaluation.sum()) < 1000:
        raise AssertionError("chronological split unexpectedly small")

    screen_rows = _candidate_metrics(frame, screen)
    screen_selected = _choose(screen_rows)
    validation_rows = _candidate_metrics(frame, validation)
    validation_lookup = {row["model"]: row for row in validation_rows}
    validation_passed = bool(
        screen_selected != "identity_early"
        and validation_lookup[screen_selected]["feasible"])
    final_model = screen_selected if validation_passed else "identity_early"

    query, *_rest = _build_query()
    event_anchor = query[
        pd.to_datetime(query.query_date).dt.date
        == pd.to_datetime(query.publication_date).dt.date
    ][["query_date", "currency", "identity_early", "t25_base"]].copy()
    event_anchor["query_date"] = pd.to_datetime(event_anchor.query_date).dt.date
    if event_anchor[["query_date", "currency"]].duplicated().any():
        raise AssertionError("publication anchor key duplicated")
    frame = frame.merge(
        event_anchor, on=["query_date", "currency"], how="left",
        suffixes=("_t4", "_query"), validate="one_to_one")
    open_rows = pd.to_datetime(frame.query_date).dt.date.to_numpy() >= EVALUATION_START
    if frame.loc[open_rows, "t25_base"].isna().any():
        raise AssertionError("T25 publication anchors missing")
    anchor_error = float(np.max(np.abs(
        frame.loc[open_rows, "identity_early_t4"]
        - frame.loc[open_rows, "identity_early_query"])))
    if anchor_error > 1e-12:
        raise AssertionError("T4/query identity anchors disagree")
    frame["identity_early"] = frame.identity_early_t4
    frame["t25_base"] = frame.t25_base.fillna(frame.identity_early)
    frame["t30_candidate"] = (
        frame.identity_early if screen_selected == "identity_early"
        else frame[screen_selected])
    frame["t30_selected"] = (
        frame.identity_early if final_model == "identity_early"
        else frame[final_model])

    development = frame[~open_rows].copy()
    evaluated = frame[open_rows].copy()
    metrics = _metrics(evaluated)
    intervals = pd.concat([
        _paired_intervals(evaluated, "t30_selected", "identity_early", "identity"),
        _paired_intervals(evaluated, "t30_selected", "t25_base", "t25"),
    ], ignore_index=True)
    primary = metrics[(metrics.model == "t30_selected") & (metrics.slice == "ALL")].iloc[0]
    brier_t25 = intervals[(intervals.comparison == "t25") & (intervals.metric == "brier")]
    brier_identity = intervals[
        (intervals.comparison == "identity") & (intervals.metric == "brier")]
    auc_identity = intervals[
        (intervals.comparison == "identity") & (intervals.metric == "auc")]
    passed = bool(
        validation_passed and final_model != "identity_early"
        and primary.brier_delta < 0.0 and primary.logloss_delta < 0.0
        and primary.ece_delta_identity <= 0.005 and primary.auc_delta_identity >= 0.02
        and primary.brier_delta_t25 < 0.0 and primary.logloss_delta_t25 < 0.0
        and primary.ece_delta_t25 <= 0.005 and primary.auc_delta_t25 >= -0.005
        and (brier_identity.ci_high < 0.0).all()
        and (brier_t25.ci_high < 0.0).all()
        and (auc_identity.ci_low > 0.0).all())

    source_files = [
        REGISTERED,
        Path("research/temperature_t30_presvo_rank_postsvo_map.py"),
        Path("research/temperature_t24_history_h20_anchor.py"),
        Path("research/temperature_t26_delayed_base_rate.py"),
        Path("results/research/temperature/t4_premarket/outputs.npz"),
        Path("research/cache/extended_features_2010_2026.npz"),
        Path("data/cbr_rates_2010_2026.json"),
    ]
    metadata = {
        "packet": "temperature-T30",
        "screen_selected": screen_selected,
        "validation_passed": validation_passed,
        "selected_model": final_model,
        "passed": passed,
        "selection_on_open_2025_2026": False,
        "open_period_previously_inspected": True,
        "fresh_independent_holdout": False,
        "splits": {
            "rank_origin": str(RANK_ORIGIN),
            "mapping_start": str(MAP_START),
            "mapping_end_exclusive": str(MAP_END),
            "screen": "2023",
            "validation": "2024",
            "evaluation": "2025-2026",
            "screen_rows": int(screen.sum()),
            "validation_rows": int(validation.sum()),
            "evaluation_rows": int(len(evaluated)),
            "screen_dates": int(frame.loc[screen, "query_date"].nunique()),
            "validation_dates": int(frame.loc[validation, "query_date"].nunique()),
            "evaluation_dates": int(evaluated.query_date.nunique()),
            "latest_screen_maturity_ord": int(frame.loc[screen, "maturity_ord"].max()),
            "screen_cutoff_ord": screen_cutoff,
            "latest_validation_maturity_ord": int(frame.loc[validation, "maturity_ord"].max()),
            "validation_cutoff_ord": validation_cutoff,
        },
        "candidate_grid": list(CANDIDATES),
        "screen": screen_rows,
        "validation": validation_rows,
        "model_details": model_details,
        "checks": {
            "rank_fit_causal": bool((
                fit_log.latest_training_maturity_ord < fit_log.cutoff_ord).all()),
            "mapping_fit_causal": bool((
                map_log.latest_mapping_maturity_ord < map_log.cutoff_ord).all()),
            "positive_platt_slopes": bool((map_log.slope > 0).all()),
            "screen_validation_disjoint": bool(not (screen & validation).any()),
            "screen_mature": bool(
                frame.loc[screen, "maturity_ord"].max() < screen_cutoff),
            "validation_mature": bool(
                frame.loc[validation, "maturity_ord"].max() < validation_cutoff),
            "publication_keys_unique": bool(not frame[
                ["query_date", "currency"]].duplicated().any()),
            "t4_query_anchor_max_abs_error": anchor_error,
            "changes_push_policy": False,
            "historical_receipts_certified": False,
            "bank_execution_validated": False,
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    return {
        "predictions": evaluated[[
            "query_date", "year", "currency", "target", "maturity_ord",
            *MODELS]],
        "development_predictions": development[[
            "query_date", "year", "currency", "target", "maturity_ord",
            "identity_early", "t30_candidate", "t30_selected", *CANDIDATES]],
        "fit_log": fit_log,
        "map_log": map_log,
        "metrics": metrics,
        "paired_bootstrap": intervals,
        "screen_validation": pd.DataFrame([
            {"stage": "screen", **row} for row in screen_rows
        ] + [{"stage": "validation", **row} for row in validation_rows]),
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(
        OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["development_predictions"].to_csv(
        OUT / "development_predictions.csv.gz", index=False, compression="gzip")
    result["fit_log"].to_csv(OUT / "fit_log.csv", index=False)
    result["map_log"].to_csv(OUT / "map_log.csv", index=False)
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    result["screen_validation"].to_csv(OUT / "screen_validation.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    overall = result["metrics"][result["metrics"].slice.eq("ALL")]
    print(json.dumps({
        "screen_selected": result["metadata"]["screen_selected"],
        "validation_passed": result["metadata"]["validation_passed"],
        "selected_model": result["metadata"]["selected_model"],
        "passed": result["metadata"]["passed"],
        "screen": result["metadata"]["screen"],
        "validation": result["metadata"]["validation"],
        "evaluation": overall.set_index("model")[[
            "auc", "average_precision", "brier", "logloss", "ece",
            "brier_delta", "logloss_delta", "auc_delta_identity",
            "brier_delta_t25", "logloss_delta_t25", "auc_delta_t25",
        ]].to_dict("index"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
