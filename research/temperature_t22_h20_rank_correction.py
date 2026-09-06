"""T22: import h20 rank information through a calibration-constrained correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from ml.data import CORRIDORS, load
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    BOOTSTRAP_DRAWS,
    CLOCKS,
    EMBARGO_DAYS,
    SCENARIOS,
    _load_snapshots,
    _probability_metrics,
    _query_grid,
    _selected_queries,
    _target_arrays,
)
from research.temperature_t21_h20_curve_head import (
    BASE_ORIGIN,
    EVAL_ORIGIN,
    _clip,
    _encode,
    _fast_auc,
    _feature_frame,
    _logit,
)


OUT = Path("results/research/temperature/t22_h20_rank_correction")
REGISTERED = Path("research/temperature_t22_h20_rank_correction_registered.md")
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)
PRIMARY = "rank_correction_selected"
MIN_BASE = 500
MIN_CALIBRATION = 200


def _alpha_name(alpha):
    return "alpha_" + f"{alpha:.3f}".replace(".", "p")


FIXED_MODELS = tuple(_alpha_name(alpha) for alpha in ALPHAS)
MODELS = ("identity_h20", PRIMARY, *FIXED_MODELS)


def _correction_probabilities(identity, residual_z, alpha):
    return _clip(expit(_logit(identity) + alpha * residual_z))


def _fit_correction(base, calibration, evaluation):
    identity = evaluation.identity_h20.to_numpy(dtype=float)
    output = {"identity_h20": identity}
    if (len(base) < MIN_BASE or len(calibration) < MIN_CALIBRATION
            or base.target.nunique() < 2 or calibration.target.nunique() < 2):
        for name in (PRIMARY, *FIXED_MODELS):
            output[name] = identity.copy()
        return output, {
            "status": "fallback_identity", "selected_alpha": 0.0,
            "features": [], "calibration_grid": [],
        }

    X_base, X_cal, X_eval, names = _encode(base, calibration, evaluation)
    ranker = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs", max_iter=5000,
        random_state=20260906)
    y_base = base.target.to_numpy(dtype=int)
    ranker.fit(X_base, y_base)
    raw_base = ranker.decision_function(X_base)
    raw_cal = ranker.decision_function(X_cal)
    raw_eval = ranker.decision_function(X_eval)

    frozen_base = _logit(base.identity_h20.to_numpy(dtype=float))
    frozen_cal = _logit(calibration.identity_h20.to_numpy(dtype=float))
    frozen_eval = _logit(identity)
    design = np.column_stack([np.ones(len(base)), frozen_base])
    intercept, slope = np.linalg.lstsq(design, raw_base, rcond=None)[0]
    residual_base = raw_base - (intercept + slope * frozen_base)
    residual_mean = float(residual_base.mean())
    residual_scale = float(residual_base.std(ddof=0))
    if not np.isfinite(residual_scale) or residual_scale < 1e-8:
        residual_scale = 1.0
    z_cal = (
        raw_cal - (intercept + slope * frozen_cal) - residual_mean
    ) / residual_scale
    z_eval = (
        raw_eval - (intercept + slope * frozen_eval) - residual_mean
    ) / residual_scale

    cal_identity = calibration.identity_h20.to_numpy(dtype=float)
    y_cal = calibration.target.to_numpy(dtype=int)
    grid = []
    for alpha in ALPHAS:
        cal_prediction = _correction_probabilities(cal_identity, z_cal, alpha)
        metrics = _probability_metrics(y_cal, cal_prediction, cal_identity)
        metrics["ece_delta"] = metrics["ece"] - metrics["baseline_ece"]
        metrics["alpha"] = alpha
        metrics["feasible"] = bool(
            metrics["brier_delta"] <= 0.001
            and metrics["logloss_delta"] <= 0.003
            and metrics["ece_delta"] <= 0.02
        )
        grid.append(metrics)
    feasible = [row for row in grid if row["feasible"]]
    if feasible:
        selected = sorted(feasible, key=lambda row: (-row["auc"], row["alpha"]))[0]
    else:
        selected = next(row for row in grid if row["alpha"] == 0.0)
    selected_alpha = float(selected["alpha"])

    for alpha in ALPHAS:
        output[_alpha_name(alpha)] = _correction_probabilities(
            identity, z_eval, alpha)
    output[PRIMARY] = output[_alpha_name(selected_alpha)].copy()
    details = {
        "status": "fit", "selected_alpha": selected_alpha,
        "features": names,
        "rank_intercept": float(ranker.intercept_[0]),
        "rank_coefficients": [float(value) for value in ranker.coef_[0]],
        "residual_intercept": float(intercept),
        "residual_slope": float(slope),
        "residual_mean": residual_mean,
        "residual_scale": residual_scale,
        "calibration_grid": grid,
    }
    return output, details


def _paired_intervals(part, scenario, clock):
    ordered = part.sort_values(["query_date", "currency"]).copy()
    dates = ordered.query_date.drop_duplicates().to_numpy()
    if len(ordered) != len(dates) * len(CORRIDORS):
        raise AssertionError("unexpected rows per date")
    y = ordered.target.to_numpy(dtype=int).reshape(len(dates), len(CORRIDORS))
    base = ordered.identity_h20.to_numpy().reshape(len(dates), len(CORRIDORS))
    primary = ordered[PRIMARY].to_numpy().reshape(len(dates), len(CORRIDORS))
    rows = []
    scenario_i = SCENARIOS.index(scenario)
    clock_i = CLOCKS.index(clock)
    for block in BLOCKS:
        rng = np.random.default_rng(
            20260907 + scenario_i * 100000 + clock_i * 1000 + block)
        blocks = int(np.ceil(len(dates) / block))
        starts = rng.integers(0, len(dates), size=(BOOTSTRAP_DRAWS, blocks))
        offsets = np.arange(block)
        sample_dates = (
            (starts[:, :, None] + offsets[None, None, :]) % len(dates)
        ).reshape(BOOTSTRAP_DRAWS, -1)[:, :len(dates)]
        brier_delta = np.empty(BOOTSTRAP_DRAWS)
        auc_delta = np.empty(BOOTSTRAP_DRAWS)
        for draw_i, sample in enumerate(sample_dates):
            ys = y[sample].ravel()
            bs = base[sample].ravel()
            ps = primary[sample].ravel()
            brier_delta[draw_i] = np.mean((ps - ys) ** 2 - (bs - ys) ** 2)
            auc_delta[draw_i] = _fast_auc(ys, ps) - _fast_auc(ys, bs)
        for metric, values in (("brier", brier_delta), ("auc", auc_delta)):
            rows.append({
                "scenario": scenario, "clock": clock, "metric": metric,
                "block_dates": block, "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, .025)),
                "ci_high": float(np.quantile(values, .975)),
            })
    return rows


def _slices(frame):
    yield "ALL", "ALL", frame
    for currency, part in frame.groupby("currency", sort=True):
        yield "currency", currency, part
    for year, part in frame.groupby("year", sort=True):
        yield "year", str(year), part
    for (currency, year), part in frame.groupby(["currency", "year"], sort=True):
        yield "currency_year", f"{currency}:{year}", part


def run():
    snapshots = _load_snapshots()
    queries = _query_grid(snapshots)
    series = load("data/cbr_rates_2010_2026.json")
    targets = _target_arrays(series)
    base_cutoff = (
        BASE_ORIGIN - pd.Timedelta(days=EMBARGO_DAYS)).date().toordinal()
    eval_cutoff = (
        EVAL_ORIGIN - pd.Timedelta(days=EMBARGO_DAYS)).date().toordinal()
    predictions = []
    fit_rows = []
    model_details = []
    for scenario in SCENARIOS:
        selected = _selected_queries(queries, snapshots, scenario)
        features = _feature_frame(selected, series, targets)
        for clock in CLOCKS:
            part = features[features.clock.eq(clock)].copy()
            date = pd.to_datetime(part.query_date)
            valid = np.isfinite(part.target)
            base = part[
                valid & (date < BASE_ORIGIN.tz_localize(None))
                & (part.maturity_ord < base_cutoff)].copy()
            calibration = part[
                valid & (date >= BASE_ORIGIN.tz_localize(None))
                & (date < EVAL_ORIGIN.tz_localize(None))
                & (part.maturity_ord < eval_cutoff)].copy()
            evaluation = part[
                valid & (date >= EVAL_ORIGIN.tz_localize(None))].copy()
            if base.empty or calibration.empty or evaluation.empty:
                raise AssertionError("empty split")
            if not (base.maturity_ord < base_cutoff).all():
                raise AssertionError("immature base label")
            if not (calibration.maturity_ord < eval_cutoff).all():
                raise AssertionError("immature calibration label")
            if pd.to_datetime(base.query_date).max() >= pd.to_datetime(
                    calibration.query_date).min():
                raise AssertionError("base/calibration overlap")
            outputs, details = _fit_correction(base, calibration, evaluation)
            saved = evaluation[[
                "scenario", "query_date", "year", "clock", "currency",
                "query_at", "source_at", "target", "p20",
            ]].copy()
            for name, values in outputs.items():
                saved[name] = values
            predictions.append(saved)
            fit_rows.append({
                "scenario": scenario, "clock": clock,
                "status": details["status"],
                "selected_alpha": details["selected_alpha"],
                "base_rows": int(len(base)),
                "calibration_rows": int(len(calibration)),
                "evaluation_rows": int(len(evaluation)),
                "latest_base_maturity_ord": int(base.maturity_ord.max()),
                "base_cutoff_ord": base_cutoff,
                "latest_calibration_maturity_ord": int(
                    calibration.maturity_ord.max()),
                "eval_cutoff_ord": eval_cutoff,
            })
            model_details.append({
                "scenario": scenario, "clock": clock, **details,
            })

    predictions = pd.concat(predictions, ignore_index=True)
    metrics = []
    intervals = []
    for (scenario, clock), part in predictions.groupby(
            ["scenario", "clock"], sort=False):
        for name in MODELS:
            for slice_name, group, sliced in _slices(part):
                result = _probability_metrics(
                    sliced.target, sliced[name], sliced.identity_h20)
                result["ece_delta"] = result["ece"] - result["baseline_ece"]
                metrics.append({
                    "scenario": scenario, "clock": clock, "h": 20,
                    "model": name, "slice": slice_name, "group": group,
                    **result,
                })
        intervals.extend(_paired_intervals(part, scenario, clock))
    metrics = pd.DataFrame(metrics)
    intervals = pd.DataFrame(intervals)

    overall = metrics[
        metrics.slice.eq("ALL")
        & metrics.model.isin(["identity_h20", PRIMARY])
    ].pivot(index=["scenario", "clock"], columns="model",
            values=["brier", "logloss", "ece", "auc", "average_precision"])
    overall.columns = ["_".join(column) for column in overall.columns]
    state = overall.reset_index()
    for metric in ["brier", "logloss", "ece", "auc", "average_precision"]:
        state[f"{metric}_delta"] = (
            state[f"{metric}_{PRIMARY}"] - state[f"{metric}_identity_h20"])
    for metric in ["brier", "auc"]:
        selected_intervals = intervals[intervals.metric.eq(metric)].groupby(
            ["scenario", "clock"]).agg(
                ci_low=("ci_low", "min"), ci_high=("ci_high", "max")
        ).add_prefix(f"{metric}_").reset_index()
        state = state.merge(selected_intervals, on=["scenario", "clock"])
    state["pass"] = (
        (state.auc_delta > 0.0) & (state.auc_ci_low > 0.0)
        & (state.brier_delta < 0.0) & (state.brier_ci_high < 0.0)
        & (state.logloss_delta < 0.0) & (state.ece_delta <= .01)
    )

    local = metrics[
        metrics.slice.eq("currency_year")
        & metrics.model.isin(["identity_h20", PRIMARY])].copy()
    local["high_ece"] = local.ece > .08
    high_ece = local.groupby("model").high_ece.sum().to_dict()
    fits = pd.DataFrame(fit_rows)
    alpha_counts = fits.selected_alpha.value_counts().sort_index().to_dict()
    source_files = [
        REGISTERED,
        Path("research/temperature_t22_h20_rank_correction.py"),
        Path("research/temperature_t21_h20_curve_head.py"),
        Path("research/temperature_t19_anytime_quality_audit.py"),
        Path("results/research/temperature/t17_spot_availability_repair/metadata.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
        Path("data/cbr_rates_2010_2026.json"),
    ]
    metadata = {
        "packet": "temperature-T22",
        "primary_candidate": PRIMARY,
        "alpha_grid": list(ALPHAS),
        "selection_on_open_period": False,
        "evaluation_rows": int(len(predictions)),
        "state_rows": int(len(state)),
        "passing_states": int(state["pass"].sum()),
        "selected_alpha_counts": {str(k): int(v) for k, v in alpha_counts.items()},
        "high_ece_currency_year_rows": {
            key: int(value) for key, value in high_ece.items()},
        "checks": {
            "all_base_labels_mature": bool((
                fits.latest_base_maturity_ord < fits.base_cutoff_ord).all()),
            "all_calibration_labels_mature": bool((
                fits.latest_calibration_maturity_ord < fits.eval_cutoff_ord).all()),
            "base_and_calibration_disjoint": True,
            "all_horizon_sources_no_later_than_query": True,
            "base_only_residualization": True,
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
    return {
        "predictions": predictions, "metrics": metrics,
        "paired_bootstrap": intervals, "state_gates": state,
        "fit_checks": fits, "model_details": model_details,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(
        OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(
        OUT / "paired_bootstrap.csv", index=False)
    result["state_gates"].to_csv(OUT / "state_gates.csv", index=False)
    result["fit_checks"].to_csv(OUT / "fit_checks.csv", index=False)
    (OUT / "model_details.json").write_text(json.dumps(
        result["model_details"], ensure_ascii=False, indent=2))
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    state = result["state_gates"]
    overall = result["metrics"][result["metrics"].slice.eq("ALL")]
    print(json.dumps({
        "passing_states": int(state["pass"].sum()),
        "states": int(len(state)),
        "selected_alpha_counts": result["metadata"]["selected_alpha_counts"],
        "mean_state_deltas": {
            column: float(state[column].mean()) for column in [
                "auc_delta", "average_precision_delta", "brier_delta",
                "logloss_delta", "ece_delta"]},
        "mean_metrics_by_model": overall.groupby("model")[[
            "auc", "average_precision", "brier", "logloss", "ece"
        ]].mean().to_dict("index"),
        "high_ece_currency_year_rows": result["metadata"][
            "high_ece_currency_year_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
