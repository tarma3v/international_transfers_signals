"""T43: frozen nonlinear annual rolling-origin h20 history expert."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ml.data import CORRIDORS
from research import temperature_t37_source_driven_h20_shrink50 as t37
from research import temperature_t38_h20_local_stability as t38
from research import temperature_t40_long_rolling_history as t40
from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS
from research.temperature_t21_h20_curve_head import _logit


OUT = Path("results/research/temperature/t43_nonlinear_history")
REGISTERED = Path("research/temperature_t43_nonlinear_history_registered.md")
CANDIDATE = "nonlinear_annual_history_h20_shadow"
SEED = 20260906
MODEL_PARAMS = {
    "loss": "log_loss",
    "max_iter": 160,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 5.0,
    "early_stopping": False,
    "random_state": SEED,
}


def _masks(base, year):
    dates = base["dates"]
    target = base["target"]
    maturity = base["maturity"]
    fit_origin = dt.date(year - 1, 1, 1)
    eval_origin = dt.date(year, 1, 1)
    fit_cutoff = (fit_origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    calibration_cutoff = (
        eval_origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    train = (
        (dates < fit_origin)
        & np.isfinite(target)
        & np.isfinite(maturity)
        & (maturity < fit_cutoff)
    )
    calibration = (
        (dates >= fit_origin)
        & (dates < eval_origin)
        & np.isfinite(target)
        & np.isfinite(maturity)
        & (maturity < calibration_cutoff)
    )
    query = (dates >= eval_origin) & (dates < dt.date(year + 1, 1, 1))
    return (
        train, calibration, query, fit_origin, eval_origin,
        fit_cutoff, calibration_cutoff,
    )


def _fit_year(base, year):
    (train, calibration, query, fit_origin, eval_origin,
     fit_cutoff, calibration_cutoff) = _masks(base, year)
    target = base["target"]
    compact = base["compact"]
    dates = base["dates"]
    maturity = base["maturity"]
    if train.sum() < 1000 or calibration.sum() < 500 or query.sum() < 500:
        raise AssertionError(f"insufficient T43 annual support for {year}")
    if (np.unique(target[train]).size != 2
            or np.unique(target[calibration]).size != 2):
        raise AssertionError(f"single-class T43 annual fit for {year}")

    raw_model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    raw_model.fit(compact[train], target[train].astype(int))
    if int(raw_model.n_iter_) != int(MODEL_PARAMS["max_iter"]):
        raise AssertionError(f"T43 estimator stopped early for {year}")
    raw_calibration = t40._clip(
        raw_model.predict_proba(compact[calibration])[:, 1])
    raw_query = t40._clip(raw_model.predict_proba(compact[query])[:, 1])
    platt = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        random_state=SEED,
    )
    platt.fit(_logit(raw_calibration)[:, None], target[calibration].astype(int))
    calibrated = t40._clip(
        platt.predict_proba(_logit(raw_query)[:, None])[:, 1])
    candidate = t37._equal_logit_blend(base["prior"][query], calibrated)

    frame = pd.DataFrame({
        "publication_date": dates[query],
        "year": year,
        "currency": base["currencies"][query],
        "target": target[query],
        "maturity_ord": maturity[query],
        "causal_prior": base["prior"][query],
        "causal_prior_count": base["prior_count"][query],
        "raw_probability": raw_query,
        "platt_probability": calibrated,
        CANDIDATE: candidate,
        "source_row": np.flatnonzero(query),
    })
    log = {
        "year": year,
        "fit_origin": str(fit_origin),
        "evaluation_origin": str(eval_origin),
        "fit_cutoff_ord": fit_cutoff,
        "calibration_cutoff_ord": calibration_cutoff,
        "training_rows": int(train.sum()),
        "calibration_rows": int(calibration.sum()),
        "query_rows": int(query.sum()),
        "latest_training_date": str(max(dates[train])),
        "latest_training_maturity_ord": int(np.max(maturity[train])),
        "latest_calibration_date": str(max(dates[calibration])),
        "latest_calibration_maturity_ord": int(np.max(maturity[calibration])),
        "feature_count": int(compact.shape[1]),
        "model_params": json.dumps(MODEL_PARAMS, sort_keys=True),
        "n_iter": int(raw_model.n_iter_),
        "platt_intercept": float(platt.intercept_[0]),
        "platt_slope": float(platt.coef_[0, 0]),
        "raw_calibration_mean": float(raw_calibration.mean()),
        "raw_query_mean": float(raw_query.mean()),
    }
    return frame, log


def _annual_predictions(base=None):
    base = t40._load_base() if base is None else base
    frames, logs = [], []
    for year in t40.YEARS:
        frame, log = _fit_year(base, year)
        frames.append(frame)
        logs.append(log)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("T43 annual publication key is not unique")
    return predictions, pd.DataFrame(logs), base


def _historical_outputs(predictions):
    aliased = predictions.copy()
    aliased[t40.CANDIDATE] = aliased[CANDIDATE]
    historical, metrics = t40._historical_metrics(aliased)
    intervals = t40._historical_bootstrap(historical)
    gates = t40._historical_gates(metrics, intervals)
    return historical, metrics, intervals, gates


def _rename_open(value):
    if not isinstance(value, pd.DataFrame):
        return value
    output = value.copy()
    output.columns = [
        str(column).replace(t40.CANDIDATE, CANDIDATE)
        for column in output.columns]
    if "model" in output:
        output["model"] = output.model.replace({t40.CANDIDATE: CANDIDATE})
    return output


def _open_outputs(predictions):
    aliased = predictions.copy()
    aliased[t40.CANDIDATE] = aliased[CANDIDATE]
    opened = t40._open_outputs(aliased)
    return {key: _rename_open(value) for key, value in opened.items()}


def _source_hashes():
    files = [
        REGISTERED,
        Path("research/temperature_t43_nonlinear_history.py"),
        Path("research/temperature_t43_nonlinear_history_audit.py"),
        Path("research/temperature_t40_long_rolling_history.py"),
        t40.FEATURE_CACHE,
        t40.LONG_DATA,
        t37.OUT / "metadata.json",
        t37.OUT / "predictions.csv.gz",
        t38.OUT / "metadata.json",
        t38.OUT / "clock_local_metrics.csv",
        t38.OUT / "pooled_local_gates.csv",
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def run():
    predictions, fit_log, _base = _annual_predictions()
    historical, metrics, intervals, gates = _historical_outputs(predictions)
    historical_gate_passed = bool(gates["pass"].all())
    open_outputs = (
        _open_outputs(predictions) if historical_gate_passed else None)
    checks = {
        "single_frozen_nonlinear_candidate": True,
        "annual_year_grid_complete": bool(
            set(predictions.year.unique()) == set(t40.YEARS)),
        "annual_currency_grid_complete": bool(
            set(predictions.currency.unique()) == set(CORRIDORS)),
        "publication_key_unique": bool(not predictions[[
            "publication_date", "currency"]].duplicated().any()),
        "probability_bounded": bool(predictions[CANDIDATE].between(
            0.0, 1.0).all()),
        "feature_count_exact_t24": bool(fit_log.feature_count.eq(41).all()),
        "fixed_iteration_count": bool(fit_log.n_iter.eq(160).all()),
        "fit_embargo_respected": bool((
            fit_log.latest_training_maturity_ord
            < fit_log.fit_cutoff_ord).all()),
        "calibration_embargo_respected": bool((
            fit_log.latest_calibration_maturity_ord
            < fit_log.calibration_cutoff_ord).all()),
        "open_not_evaluated_if_historical_failed": bool(
            historical_gate_passed or open_outputs is None),
    }
    if open_outputs is not None:
        checks["non_history_exact_t37"] = bool(
            open_outputs["non_history_exact"])
    metadata = {
        "packet": "temperature-T43",
        "candidate": CANDIDATE,
        "raw_estimator": "HistGradientBoostingClassifier",
        "model_params": MODEL_PARAMS,
        "calibrator": "previous-year Platt logistic on raw logit",
        "blend": "fixed equal log-odds with causal prior",
        "historical_screen": "2019-2022",
        "historical_validation": "2023-2024",
        "open_diagnostic": "2025-2026 only after historical pass",
        "historical_gate_passed": historical_gate_passed,
        "open_evaluated": open_outputs is not None,
        "open_repair_passed": bool(
            open_outputs and open_outputs["open_repair_passed"]),
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
        "changes_push_policy": False,
        "changes_expected_future_bps": False,
        "changes_runtime_router": False,
        "checks": checks,
        "source_sha256": _source_hashes(),
    }
    result = {
        "publication_predictions": predictions,
        "annual_fit_log": fit_log,
        "historical_metrics": metrics,
        "historical_paired_bootstrap": intervals,
        "historical_gates": gates,
        "metadata": metadata,
    }
    if open_outputs is not None:
        result.update(open_outputs)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["publication_predictions"].to_csv(
        OUT / "publication_predictions.csv.gz", index=False,
        compression="gzip")
    for key in (
        "annual_fit_log", "historical_metrics",
        "historical_paired_bootstrap", "historical_gates",
    ):
        result[key].to_csv(OUT / f"{key}.csv", index=False)
    if result["metadata"]["open_evaluated"]:
        for key in (
            "predictions", "metrics", "reliability", "paired_bootstrap",
            "state_summary", "component_metrics", "pooled_metrics",
            "pooled_paired_bootstrap", "pooled_gate",
            "pairwise_state_metrics", "pairwise_pooled_metrics",
            "clock_local_metrics", "pooled_local_metrics",
            "local_paired_bootstrap", "pooled_local_gates", "failure_summary",
        ):
            suffix = ".csv.gz" if key == "predictions" else ".csv"
            result[key].to_csv(
                OUT / f"{key}{suffix}", index=False,
                compression="gzip" if suffix.endswith(".gz") else None)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "historical_gates": result["historical_gates"].to_dict("records"),
        "historical_gate_passed": result["metadata"][
            "historical_gate_passed"],
        "open_evaluated": result["metadata"]["open_evaluated"],
        "production_promoted": False,
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
