"""T44: disjoint mature quality gate for the nonlinear h20 history expert."""
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
from research import temperature_t43_nonlinear_history as t43
from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS
from research.temperature_t21_h20_curve_head import _logit


OUT = Path("results/research/temperature/t44_calibration_quality_gate")
REGISTERED = Path(
    "research/temperature_t44_calibration_quality_gate_registered.md")
CANDIDATE = "quality_gated_nonlinear_history_h20_shadow"
GATE_CANDIDATE = "early_quality_gate_candidate"
SPLIT_MONTH = 7
MIN_HALF_ROWS = 400


def _metric_row(part, candidate=GATE_CANDIDATE):
    aliased = part.copy()
    aliased[t40.CANDIDATE] = aliased[candidate]
    return t40._metric_row(aliased)


def _gate_metrics(frame, year):
    rows = [{
        "year": year,
        "slice": "pooled",
        "group": "all",
        **_metric_row(frame),
    }]
    for currency, part in frame.groupby("currency", sort=True):
        rows.append({
            "year": year,
            "slice": "currency",
            "group": str(currency),
            **_metric_row(part),
        })
    output = pd.DataFrame(rows)
    output["noninferior"] = (
        (output.brier_delta <= 0.001)
        & (output.logloss_delta <= 0.003)
        & (output.ece_delta <= 0.01)
        & (output.auc_delta >= -0.005)
    )
    return output


def _gate_decision(metrics, early_platt_slope):
    pooled = metrics[metrics.slice.eq("pooled")].iloc[0]
    local = metrics[metrics.slice.eq("currency")]
    return bool(
        early_platt_slope > 0.0
        and pooled.brier_delta < 0.0
        and pooled.logloss_delta < 0.0
        and pooled.auc_delta > 0.0
        and pooled.ece_delta <= 0.01
        and len(local) == len(CORRIDORS)
        and local.noninferior.all()
    )


def _fit_year(base, year):
    (train, calibration, query, fit_origin, eval_origin,
     fit_cutoff, calibration_cutoff) = t43._masks(base, year)
    dates = base["dates"]
    target = base["target"]
    maturity = base["maturity"]
    compact = base["compact"]
    currencies = base["currencies"]
    split_origin = dt.date(year - 1, SPLIT_MONTH, 1)
    split_cutoff = (
        split_origin - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    early_calibration = (
        calibration
        & (dates < split_origin)
        & (maturity < split_cutoff)
    )
    quality_gate = calibration & (dates >= split_origin)
    if (train.sum() < 1000 or calibration.sum() < 500
            or query.sum() < 500):
        raise AssertionError(f"insufficient T44 annual support for {year}")
    if (early_calibration.sum() < MIN_HALF_ROWS
            or quality_gate.sum() < MIN_HALF_ROWS):
        raise AssertionError(f"insufficient T44 half-year support for {year}")
    for label, mask in (
            ("train", train), ("early calibration", early_calibration),
            ("quality gate", quality_gate), ("full calibration", calibration)):
        if np.unique(target[mask]).size != 2:
            raise AssertionError(f"single-class T44 {label} for {year}")

    raw_model = HistGradientBoostingClassifier(**t43.MODEL_PARAMS)
    raw_model.fit(compact[train], target[train].astype(int))
    if int(raw_model.n_iter_) != int(t43.MODEL_PARAMS["max_iter"]):
        raise AssertionError(f"T44 estimator stopped early for {year}")

    raw_early = t40._clip(
        raw_model.predict_proba(compact[early_calibration])[:, 1])
    raw_gate = t40._clip(
        raw_model.predict_proba(compact[quality_gate])[:, 1])
    raw_full = t40._clip(
        raw_model.predict_proba(compact[calibration])[:, 1])
    raw_query = t40._clip(
        raw_model.predict_proba(compact[query])[:, 1])

    early_platt = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=3000,
        random_state=t43.SEED)
    early_platt.fit(
        _logit(raw_early)[:, None], target[early_calibration].astype(int))
    gate_calibrated = t40._clip(early_platt.predict_proba(
        _logit(raw_gate)[:, None])[:, 1])
    gate_candidate = t37._equal_logit_blend(
        base["prior"][quality_gate], gate_calibrated)
    gate_frame = pd.DataFrame({
        "publication_date": dates[quality_gate],
        "currency": currencies[quality_gate],
        "target": target[quality_gate],
        "causal_prior": base["prior"][quality_gate],
        GATE_CANDIDATE: gate_candidate,
    })
    gate_metrics = _gate_metrics(gate_frame, year)
    gate_open = _gate_decision(
        gate_metrics, float(early_platt.coef_[0, 0]))

    full_platt = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=3000,
        random_state=t43.SEED)
    full_platt.fit(
        _logit(raw_full)[:, None], target[calibration].astype(int))
    full_calibrated = t40._clip(full_platt.predict_proba(
        _logit(raw_query)[:, None])[:, 1])
    nonlinear = t37._equal_logit_blend(
        base["prior"][query], full_calibrated)
    candidate = nonlinear if gate_open else base["prior"][query].copy()

    query_rows = np.flatnonzero(query)
    frame = pd.DataFrame({
        "publication_date": dates[query],
        "year": year,
        "currency": currencies[query],
        "target": target[query],
        "maturity_ord": maturity[query],
        "causal_prior": base["prior"][query],
        "causal_prior_count": base["prior_count"][query],
        "raw_probability": raw_query,
        "platt_probability": full_calibrated,
        "gate_open": gate_open,
        CANDIDATE: candidate,
        "source_row": query_rows,
    })
    pooled = gate_metrics[gate_metrics.slice.eq("pooled")].iloc[0]
    local = gate_metrics[gate_metrics.slice.eq("currency")]
    log = {
        "year": year,
        "fit_origin": str(fit_origin),
        "split_origin": str(split_origin),
        "evaluation_origin": str(eval_origin),
        "fit_cutoff_ord": fit_cutoff,
        "split_cutoff_ord": split_cutoff,
        "calibration_cutoff_ord": calibration_cutoff,
        "training_rows": int(train.sum()),
        "early_calibration_rows": int(early_calibration.sum()),
        "quality_gate_rows": int(quality_gate.sum()),
        "full_calibration_rows": int(calibration.sum()),
        "query_rows": int(query.sum()),
        "latest_training_date": str(max(dates[train])),
        "latest_training_maturity_ord": int(np.max(maturity[train])),
        "latest_early_calibration_date": str(max(dates[early_calibration])),
        "latest_early_calibration_maturity_ord": int(
            np.max(maturity[early_calibration])),
        "latest_quality_gate_date": str(max(dates[quality_gate])),
        "latest_quality_gate_maturity_ord": int(np.max(maturity[quality_gate])),
        "latest_full_calibration_date": str(max(dates[calibration])),
        "latest_full_calibration_maturity_ord": int(
            np.max(maturity[calibration])),
        "feature_count": int(compact.shape[1]),
        "model_params": json.dumps(t43.MODEL_PARAMS, sort_keys=True),
        "n_iter": int(raw_model.n_iter_),
        "early_platt_intercept": float(early_platt.intercept_[0]),
        "early_platt_slope": float(early_platt.coef_[0, 0]),
        "full_platt_intercept": float(full_platt.intercept_[0]),
        "full_platt_slope": float(full_platt.coef_[0, 0]),
        "gate_brier_delta": float(pooled.brier_delta),
        "gate_logloss_delta": float(pooled.logloss_delta),
        "gate_ece_delta": float(pooled.ece_delta),
        "gate_auc_delta": float(pooled.auc_delta),
        "gate_local_noninferior": int(local.noninferior.sum()),
        "gate_local_groups": int(len(local)),
        "gate_open": gate_open,
        "rejected_exact_prior": bool(
            gate_open or np.array_equal(candidate, base["prior"][query])),
    }
    return frame, log, gate_metrics


def _annual_predictions(base=None):
    base = t40._load_base() if base is None else base
    frames, logs, gate_frames = [], [], []
    for year in t40.YEARS:
        frame, log, gate = _fit_year(base, year)
        frames.append(frame)
        logs.append(log)
        gate_frames.append(gate)
    predictions = pd.concat(frames, ignore_index=True)
    gate_metrics = pd.concat(gate_frames, ignore_index=True)
    if predictions[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("T44 annual publication key is not unique")
    return predictions, pd.DataFrame(logs), gate_metrics, base


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
    return {
        key: _rename_open(value)
        for key, value in t40._open_outputs(aliased).items()
    }


def _source_hashes():
    files = [
        REGISTERED,
        Path("research/temperature_t44_calibration_quality_gate.py"),
        Path("research/temperature_t44_calibration_quality_gate_audit.py"),
        Path("research/temperature_t43_nonlinear_history.py"),
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
    predictions, fit_log, gate_metrics, _base = _annual_predictions()
    historical, metrics, intervals, gates = _historical_outputs(predictions)
    historical_gate_passed = bool(gates["pass"].all())
    open_outputs = _open_outputs(predictions) if historical_gate_passed else None
    checks = {
        "single_frozen_candidate": True,
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
        "early_half_support": bool(
            fit_log.early_calibration_rows.ge(MIN_HALF_ROWS).all()),
        "gate_half_support": bool(
            fit_log.quality_gate_rows.ge(MIN_HALF_ROWS).all()),
        "fit_embargo_respected": bool((
            fit_log.latest_training_maturity_ord
            < fit_log.fit_cutoff_ord).all()),
        "early_calibration_embargo_respected": bool((
            fit_log.latest_early_calibration_maturity_ord
            < fit_log.split_cutoff_ord).all()),
        "quality_gate_embargo_respected": bool((
            fit_log.latest_quality_gate_maturity_ord
            < fit_log.calibration_cutoff_ord).all()),
        "full_calibration_embargo_respected": bool((
            fit_log.latest_full_calibration_maturity_ord
            < fit_log.calibration_cutoff_ord).all()),
        "rejected_years_exact_prior": bool(
            fit_log.rejected_exact_prior.all()),
        "open_not_evaluated_if_historical_failed": bool(
            historical_gate_passed or open_outputs is None),
    }
    if open_outputs is not None:
        checks["non_history_exact_t37"] = bool(
            open_outputs["non_history_exact"])
    metadata = {
        "packet": "temperature-T44",
        "candidate": CANDIDATE,
        "raw_estimator": "HistGradientBoostingClassifier",
        "model_params": t43.MODEL_PARAMS,
        "calibrator": "early-half Platt for gate; full prior-year Platt for query",
        "gate_validation": "mature July-December of prior year",
        "split_month": SPLIT_MONTH,
        "min_half_rows": MIN_HALF_ROWS,
        "blend": "fixed equal log-odds with causal prior when gate opens",
        "historical_screen": "2019-2022",
        "historical_validation": "2023-2024",
        "open_diagnostic": "2025-2026 only after historical pass",
        "gate_open_years": [
            int(year) for year in fit_log.loc[fit_log.gate_open, "year"]],
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
        "quality_gate_metrics": gate_metrics,
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
        "annual_fit_log", "quality_gate_metrics", "historical_metrics",
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
        "gate_open_years": result["metadata"]["gate_open_years"],
        "historical_gates": result["historical_gates"].to_dict("records"),
        "historical_gate_passed": result["metadata"][
            "historical_gate_passed"],
        "open_evaluated": result["metadata"]["open_evaluated"],
        "production_promoted": False,
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
