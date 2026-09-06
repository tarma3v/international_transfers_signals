"""T21: cross-horizon h20 discrimination heads trained before 2025."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    BOOTSTRAP_DRAWS,
    CLOCKS,
    EMBARGO_DAYS,
    SCENARIOS,
    _current_indices,
    _field,
    _freshness,
    _load_snapshots,
    _probability_metrics,
    _query_grid,
    _selected_queries,
    _target_arrays,
)


OUT = Path("results/research/temperature/t21_h20_curve_head")
REGISTERED = Path("research/temperature_t21_h20_curve_head_registered.md")
BASE_ORIGIN = pd.Timestamp("2024-09-01T00:00:00+03:00")
EVAL_ORIGIN = pd.Timestamp("2025-01-01T00:00:00+03:00")
MODELS = ("identity_h20", "curve_logit", "curve_hgb_platt", "curve_extra_platt")
PRIMARY = "curve_hgb_platt"
MIN_BASE = 500
MIN_CALIBRATION = 200


def _clip(values):
    return np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)


def _logit(values):
    values = _clip(values)
    return np.log(values / (1.0 - values))


def _maturity_and_target(selected, series, targets):
    current = _current_indices(selected, series)
    target = np.full(len(selected), np.nan)
    maturity = np.full(len(selected), np.nan)
    for currency in CORRIDORS:
        mask = selected.currency.eq(currency).to_numpy()
        fav, _benefit, mature = targets[(currency, 20)]
        indices = current[mask]
        target[mask] = fav[indices]
        maturity[mask] = mature[indices]
    return target, maturity


def _feature_frame(selected, series, targets):
    result = selected[[
        "scenario", "query_date", "year", "clock", "currency", "query_at",
    ]].copy()
    result["target"], result["maturity_ord"] = _maturity_and_target(
        selected, series, targets)
    probability_sources = []
    benefit_sources = []
    for h in HORIZONS:
        result[f"p{h}"] = pd.to_numeric(
            selected[f"snapshot_probability_h{h}"], errors="coerce")
        result[f"b{h}"] = pd.to_numeric(
            selected[f"snapshot_expected_future_bps_h{h}"], errors="coerce")
        p_source = pd.to_datetime(
            _field(selected, "source_at", h, "snapshot_source_at"), utc=True)
        b_source = pd.to_datetime(
            _field(selected, "benefit_source_at", h, "snapshot_source_at"),
            utc=True)
        if (p_source > selected.query_at).any():
            raise AssertionError(f"future probability source h{h}")
        if (b_source > selected.query_at).any():
            raise AssertionError(f"future benefit source h{h}")
        probability_sources.append(p_source)
        benefit_sources.append(b_source)

    h20_source = probability_sources[-1]
    h20_kind = _field(
        selected, "source_kind", 20, "snapshot_source_kind").astype(str)
    h20_phase = _field(selected, "phase", 20, "snapshot_phase").astype(str)
    h20_confidence = _field(
        selected, "confidence", 20, "snapshot_confidence").astype(str)
    age = (selected.query_at - h20_source).dt.total_seconds() / 60.0
    freshness = _freshness(age, h20_kind)
    result["age_minutes"] = age
    result["source_at"] = h20_source
    result["source_kind"] = h20_kind.to_numpy()
    result["phase"] = h20_phase.to_numpy()
    result["confidence"] = h20_confidence.to_numpy()
    result["freshness"] = freshness
    result["regime"] = (
        result.phase.astype(str) + "|" + result.source_kind.astype(str)
        + "|" + result.confidence.astype(str) + "|"
        + result.freshness.astype(str)
    )

    pcols = [f"p{h}" for h in HORIZONS]
    bcols = [f"b{h}" for h in HORIZONS]
    result["p_short_mean"] = result[["p1", "p3", "p5"]].mean(axis=1)
    result["p_curve_slope"] = result.p20 - result.p1
    result["p_mid_long"] = result.p10 - result.p20
    result["p_dispersion"] = result[pcols].std(axis=1)
    result["b_short_mean"] = result[["b1", "b3", "b5"]].mean(axis=1)
    result["b_curve_slope"] = result.b20 - result.b1
    result["b_mid_long"] = result.b10 - result.b20
    result["b_dispersion"] = result[bcols].std(axis=1)
    group = result.groupby(["query_date", "clock"], sort=False)
    result["p20_relative"] = result.p20 - group.p20.transform("mean")
    result["b20_relative"] = result.b20 - group.b20.transform("mean")
    result["p5_relative"] = result.p5 - group.p5.transform("mean")
    result["b5_relative"] = result.b5 - group.b5.transform("mean")
    result["identity_h20"] = _clip(result.p20)
    return result


NUMERIC = [
    "p1", "p3", "p5", "p10", "p20",
    "b1", "b3", "b5", "b10", "b20",
    "p_short_mean", "p_curve_slope", "p_mid_long", "p_dispersion",
    "b_short_mean", "b_curve_slope", "b_mid_long", "b_dispersion",
    "p20_relative", "b20_relative", "p5_relative", "b5_relative",
    "age_minutes",
]


def _encode(base, calibration, evaluation):
    medians = base[NUMERIC].median()
    scaler = StandardScaler()
    base_numeric = scaler.fit_transform(base[NUMERIC].fillna(medians))
    cal_numeric = scaler.transform(calibration[NUMERIC].fillna(medians))
    eval_numeric = scaler.transform(evaluation[NUMERIC].fillna(medians))
    train_categories = {
        "currency": sorted(base.currency.astype(str).unique()),
        "regime": sorted(base.regime.astype(str).unique()),
    }
    base_parts = [base_numeric]
    cal_parts = [cal_numeric]
    eval_parts = [eval_numeric]
    names = list(NUMERIC)
    for column, categories in train_categories.items():
        for value in categories[1:]:
            base_parts.append(
                (base[column].astype(str) == value).to_numpy(dtype=float)[:, None])
            cal_parts.append(
                (calibration[column].astype(str) == value).to_numpy(dtype=float)[:, None])
            eval_parts.append(
                (evaluation[column].astype(str) == value).to_numpy(dtype=float)[:, None])
            names.append(f"{column}={value}")
    return (
        np.column_stack(base_parts), np.column_stack(cal_parts),
        np.column_stack(eval_parts), names,
    )


def _platt(raw_calibration, y_calibration, raw_evaluation):
    mapper = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=3000,
        random_state=20260906)
    mapper.fit(_logit(raw_calibration)[:, None], y_calibration)
    prediction = mapper.predict_proba(_logit(raw_evaluation)[:, None])[:, 1]
    return prediction, {
        "platt_intercept": float(mapper.intercept_[0]),
        "platt_slope": float(mapper.coef_[0, 0]),
    }


def _fit_candidates(base, calibration, evaluation):
    output = {"identity_h20": evaluation.identity_h20.to_numpy(dtype=float)}
    details = {"identity_h20": {"status": "identity", "features": []}}
    if (len(base) < MIN_BASE or len(calibration) < MIN_CALIBRATION
            or base.target.nunique() < 2 or calibration.target.nunique() < 2):
        for name in MODELS[1:]:
            output[name] = output["identity_h20"].copy()
            details[name] = {"status": "fallback_identity", "features": []}
        return output, details

    combined = pd.concat([base, calibration], ignore_index=True)
    X_base, X_cal, X_eval, names = _encode(base, calibration, evaluation)
    # _encode also returns a calibration matrix.  A single placeholder row
    # keeps sklearn's transform contract satisfied; this matrix is not used.
    X_combined, _unused, X_eval_combined, names_combined = _encode(
        combined, calibration.iloc[:1], evaluation)
    linear = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs", max_iter=5000,
        random_state=20260906)
    linear.fit(X_combined, combined.target.to_numpy(dtype=int))
    output["curve_logit"] = linear.predict_proba(X_eval_combined)[:, 1]
    details["curve_logit"] = {
        "status": "fit", "features": names_combined,
        "intercept": float(linear.intercept_[0]),
        "coefficients": [float(x) for x in linear.coef_[0]],
    }

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=120, max_depth=2,
        min_samples_leaf=60, l2_regularization=10.0,
        random_state=20260906)
    hgb.fit(X_base, base.target.to_numpy(dtype=int))
    raw_cal = hgb.predict_proba(X_cal)[:, 1]
    raw_eval = hgb.predict_proba(X_eval)[:, 1]
    output["curve_hgb_platt"], mapper = _platt(
        raw_cal, calibration.target.to_numpy(dtype=int), raw_eval)
    details["curve_hgb_platt"] = {
        "status": "fit", "features": names, **mapper}

    extra = ExtraTreesClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=40,
        max_features=0.8, class_weight=None, n_jobs=1,
        random_state=20260906)
    extra.fit(X_base, base.target.to_numpy(dtype=int))
    raw_cal = extra.predict_proba(X_cal)[:, 1]
    raw_eval = extra.predict_proba(X_eval)[:, 1]
    output["curve_extra_platt"], mapper = _platt(
        raw_cal, calibration.target.to_numpy(dtype=int), raw_eval)
    details["curve_extra_platt"] = {
        "status": "fit", "features": names,
        "feature_importances": [float(x) for x in extra.feature_importances_],
        **mapper,
    }
    return output, details


def _fast_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = int(len(y) - n_pos)
    if not n_pos or not n_neg:
        return np.nan
    ranks = rankdata(score, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


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
            20260906 + scenario_i * 100000 + clock_i * 1000 + block)
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
        rows.extend([
            {
                "scenario": scenario, "clock": clock, "metric": "brier",
                "block_dates": block, "mean_delta": float(brier_delta.mean()),
                "ci_low": float(np.quantile(brier_delta, .025)),
                "ci_high": float(np.quantile(brier_delta, .975)),
            },
            {
                "scenario": scenario, "clock": clock, "metric": "auc",
                "block_dates": block, "mean_delta": float(auc_delta.mean()),
                "ci_low": float(np.quantile(auc_delta, .025)),
                "ci_high": float(np.quantile(auc_delta, .975)),
            },
        ])
    return rows


def _slices(frame):
    yield "ALL", "ALL", frame
    for currency, part in frame.groupby("currency", sort=True):
        yield "currency", currency, part
    for year, part in frame.groupby("year", sort=True):
        yield "year", str(year), part
    for (currency, year), part in frame.groupby(
            ["currency", "year"], sort=True):
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
            outputs, details = _fit_candidates(base, calibration, evaluation)
            saved = evaluation[[
                "scenario", "query_date", "year", "clock", "currency",
                "query_at", "source_at", "target", "p20",
            ]].copy()
            for name, values in outputs.items():
                saved[name] = values
            predictions.append(saved)
            for name, detail in details.items():
                fit_rows.append({
                    "scenario": scenario, "clock": clock, "model": name,
                    "status": detail["status"], "base_rows": int(len(base)),
                    "calibration_rows": int(len(calibration)),
                    "evaluation_rows": int(len(evaluation)),
                    "latest_base_maturity_ord": int(base.maturity_ord.max()),
                    "base_cutoff_ord": base_cutoff,
                    "latest_calibration_maturity_ord": int(
                        calibration.maturity_ord.max()),
                    "eval_cutoff_ord": eval_cutoff,
                })
                model_details.append({
                    "scenario": scenario, "clock": clock, "model": name,
                    **detail,
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
    overall.columns = ["_".join(col) for col in overall.columns]
    state = overall.reset_index()
    state["brier_delta"] = state[f"brier_{PRIMARY}"] - state.brier_identity_h20
    state["logloss_delta"] = (
        state[f"logloss_{PRIMARY}"] - state.logloss_identity_h20)
    state["ece_delta"] = state[f"ece_{PRIMARY}"] - state.ece_identity_h20
    state["auc_delta"] = state[f"auc_{PRIMARY}"] - state.auc_identity_h20
    state["ap_delta"] = (
        state[f"average_precision_{PRIMARY}"]
        - state.average_precision_identity_h20)
    for metric in ["brier", "auc"]:
        selected_intervals = intervals[intervals.metric.eq(metric)].groupby(
            ["scenario", "clock"]).agg(
                ci_low=("ci_low", "min"), ci_high=("ci_high", "max"),
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
    source_files = [
        REGISTERED,
        Path("research/temperature_t21_h20_curve_head.py"),
        Path("research/temperature_t19_anytime_quality_audit.py"),
        Path("results/research/temperature/t17_spot_availability_repair/metadata.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
        Path("data/cbr_rates_2010_2026.json"),
    ]
    fit_checks = pd.DataFrame(fit_rows)
    metadata = {
        "packet": "temperature-T21",
        "primary_candidate": PRIMARY,
        "selection_on_open_period": False,
        "evaluation_rows": int(len(predictions)),
        "state_rows": int(len(state)),
        "passing_states": int(state["pass"].sum()),
        "high_ece_currency_year_rows": {
            key: int(value) for key, value in high_ece.items()},
        "checks": {
            "all_base_labels_mature": bool((
                fit_checks.latest_base_maturity_ord
                < fit_checks.base_cutoff_ord).all()),
            "all_calibration_labels_mature": bool((
                fit_checks.latest_calibration_maturity_ord
                < fit_checks.eval_cutoff_ord).all()),
            "base_and_calibration_disjoint": True,
            "all_horizon_sources_no_later_than_query": True,
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
        "fit_checks": fit_checks, "model_details": model_details,
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
    metrics = result["metrics"]
    overall = metrics[metrics.slice.eq("ALL")]
    print(json.dumps({
        "passing_states": int(state["pass"].sum()),
        "states": int(len(state)),
        "mean_state_deltas": {
            column: float(state[column].mean()) for column in [
                "auc_delta", "ap_delta", "brier_delta", "logloss_delta",
                "ece_delta"]},
        "mean_metrics_by_model": overall.groupby("model")[[
            "auc", "average_precision", "brier", "logloss", "ece"
        ]].mean().to_dict("index"),
        "high_ece_currency_year_rows": result["metadata"][
            "high_ece_currency_year_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
