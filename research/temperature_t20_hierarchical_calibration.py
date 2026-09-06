"""T20: pre-2025 hierarchical recalibration of the frozen anytime probability."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.data import CORRIDORS, load
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    CLOCKS,
    EMBARGO_DAYS,
    SCENARIOS,
    _current_indices,
    _load_snapshots,
    _moving_block_interval,
    _probability_metrics,
    _query_grid,
    _scored_horizon,
    _selected_queries,
    _target_arrays,
)


OUT = Path("results/research/temperature/t20_hierarchical_calibration")
REGISTERED = Path("research/temperature_t20_hierarchical_calibration_registered.md")
HORIZONS = (5, 10, 20)
FIT_ORIGIN = pd.Timestamp("2025-01-01T00:00:00+03:00")
MODELS = (
    "identity", "global_platt", "fixed_logit_shrink_80",
    "hierarchical_beta",
)
MIN_TRAIN = 200


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-5, 1.0 - 1e-5)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _regime(frame):
    return (
        frame.phase.astype(str) + "|" + frame.source_kind.astype(str)
        + "|" + frame.confidence.astype(str) + "|"
        + frame.freshness.astype(str)
    )


def _maturity_ord(selected, series, h):
    current = _current_indices(selected, series)
    result = np.full(len(selected), np.nan)
    for currency in CORRIDORS:
        mask = selected.currency.eq(currency).to_numpy()
        item = series[currency]
        indices = current[mask] + h
        valid = indices < len(item.dates)
        values = np.full(mask.sum(), np.nan)
        values[valid] = [item.dates[i].toordinal() for i in indices[valid]]
        result[mask] = values
    return result


def _dummy_columns(train_values, eval_values, prefix):
    categories = sorted(pd.Series(train_values).astype(str).unique())
    columns_train, columns_eval, names = [], [], []
    for value in categories[1:]:
        columns_train.append((pd.Series(train_values).astype(str) == value)
                             .to_numpy(dtype=float))
        columns_eval.append((pd.Series(eval_values).astype(str) == value)
                            .to_numpy(dtype=float))
        names.append(f"{prefix}={value}")
    return columns_train, columns_eval, names


def _design(train, evaluation, kind):
    p_train = _clip(train.probability)
    p_eval = _clip(evaluation.probability)
    if kind == "global_platt":
        return (
            _logit(p_train)[:, None],
            _logit(p_eval)[:, None],
            ["base_logit"],
        )

    logp_train = np.log(p_train)
    logp_eval = np.log(p_eval)
    nlog1_train = -np.log1p(-p_train)
    nlog1_eval = -np.log1p(-p_eval)
    base_logit_train = _logit(p_train)
    base_logit_eval = _logit(p_eval)
    train_columns = [logp_train, nlog1_train]
    eval_columns = [logp_eval, nlog1_eval]
    names = ["log_p", "neg_log_1mp"]

    currency_train, currency_eval, currency_names = _dummy_columns(
        train.currency, evaluation.currency, "currency")
    train_columns.extend(currency_train)
    eval_columns.extend(currency_eval)
    names.extend(currency_names)
    for train_dummy, eval_dummy, name in zip(
            currency_train, currency_eval, currency_names):
        train_columns.append(train_dummy * base_logit_train)
        eval_columns.append(eval_dummy * base_logit_eval)
        names.append(f"{name}:base_logit")

    regime_train, regime_eval, regime_names = _dummy_columns(
        _regime(train), _regime(evaluation), "regime")
    train_columns.extend(regime_train)
    eval_columns.extend(regime_eval)
    names.extend(regime_names)
    return np.column_stack(train_columns), np.column_stack(eval_columns), names


def _fit_candidate(train, evaluation, kind):
    base = _clip(evaluation.probability)
    if kind == "identity":
        return base, {"status": "identity", "coefficients": {}}
    if kind == "fixed_logit_shrink_80":
        return _sigmoid(0.8 * _logit(base)), {
            "status": "fixed", "coefficients": {"base_logit": 0.8}}
    if len(train) < MIN_TRAIN or train.target.nunique() < 2:
        return base, {
            "status": "fallback_identity", "reason": "insufficient_train",
            "train_rows": int(len(train)), "coefficients": {},
        }
    X_train, X_eval, names = _design(train, evaluation, kind)
    c = 1.0 if kind == "global_platt" else 0.05
    model = LogisticRegression(
        C=c, penalty="l2", solver="lbfgs", max_iter=5000,
        random_state=20260906,
    )
    model.fit(X_train, train.target.to_numpy(dtype=int))
    prediction = model.predict_proba(X_eval)[:, 1]
    coefficients = {
        "intercept": float(model.intercept_[0]),
        **{name: float(value) for name, value in zip(
            names, model.coef_[0])},
    }
    return prediction, {
        "status": "fit", "train_rows": int(len(train)),
        "train_positive_rate": float(train.target.mean()),
        "coefficients": coefficients,
    }


def _slices(frame):
    yield "ALL", "ALL", frame
    for currency, part in frame.groupby("currency", sort=True):
        yield "currency", currency, part
    for (currency, year), part in frame.groupby(
            ["currency", "year"], sort=True):
        yield "currency_year", f"{currency}:{year}", part


def _bootstrap(part, scenario, clock, h):
    work = part.copy()
    work["delta"] = (
        (work.hierarchical_beta - work.target) ** 2
        - (work.identity - work.target) ** 2
    )
    daily = work.groupby("query_date").delta.mean()
    rows = []
    scenario_i = SCENARIOS.index(scenario)
    clock_i = CLOCKS.index(clock)
    for block in BLOCKS:
        seed = 20260906 + scenario_i * 100000 + clock_i * 1000 + h * 10 + block
        low, high, probability_better = _moving_block_interval(
            daily.to_numpy(), block, seed)
        rows.append({
            "scenario": scenario, "clock": clock, "h": h,
            "block_dates": block, "n_dates": int(len(daily)),
            "mean_brier_delta": float(daily.mean()),
            "ci_low": low, "ci_high": high,
            "probability_hierarchical_better": probability_better,
        })
    return rows


def run():
    snapshots = _load_snapshots()
    queries = _query_grid(snapshots)
    series = load("data/cbr_rates_2010_2026.json")
    targets = _target_arrays(series)
    selected_by_scenario = {
        scenario: _selected_queries(queries, snapshots, scenario)
        for scenario in SCENARIOS
    }
    fit_cutoff = (FIT_ORIGIN - pd.Timedelta(days=EMBARGO_DAYS)).date().toordinal()

    prediction_rows = []
    coefficient_rows = []
    fit_checks = []
    for scenario in SCENARIOS:
        selected = selected_by_scenario[scenario]
        for h in HORIZONS:
            scored = _scored_horizon(selected, series, targets, h)
            scored["maturity_ord"] = _maturity_ord(selected, series, h)
            scored["year"] = pd.to_datetime(scored.query_date).dt.year
            for clock in CLOCKS:
                part = scored[scored.clock.eq(clock)].copy()
                train = part[
                    (pd.to_datetime(part.query_date) < FIT_ORIGIN.tz_localize(None))
                    & (part.maturity_ord < fit_cutoff)
                    & np.isfinite(part.target)
                    & np.isfinite(part.probability)
                ].copy()
                evaluation = part[
                    (pd.to_datetime(part.query_date) >= FIT_ORIGIN.tz_localize(None))
                    & np.isfinite(part.target)
                    & np.isfinite(part.probability)
                ].copy()
                if train.empty or evaluation.empty:
                    raise AssertionError("empty train/evaluation state")
                if not (train.maturity_ord < fit_cutoff).all():
                    raise AssertionError("immature training label")
                output = evaluation[[
                    "scenario", "query_date", "year", "clock", "currency",
                    "query_at", "source_at", "source_kind", "phase",
                    "confidence", "freshness", "target", "probability",
                ]].copy()
                for model_name in MODELS:
                    prediction, fit_info = _fit_candidate(
                        train, evaluation, model_name)
                    output[model_name] = prediction
                    fit_checks.append({
                        "scenario": scenario, "clock": clock, "h": h,
                        "model": model_name,
                        "train_rows": int(len(train)),
                        "evaluation_rows": int(len(evaluation)),
                        "latest_train_maturity_ord": int(train.maturity_ord.max()),
                        "fit_cutoff_ord": int(fit_cutoff),
                        "status": fit_info["status"],
                        "reason": fit_info.get("reason", ""),
                    })
                    for feature, value in fit_info["coefficients"].items():
                        coefficient_rows.append({
                            "scenario": scenario, "clock": clock, "h": h,
                            "model": model_name, "feature": feature,
                            "coefficient": value,
                        })
                output["h"] = h
                prediction_rows.append(output)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = []
    bootstrap = []
    reliability = []
    for (scenario, clock, h), part in predictions.groupby(
            ["scenario", "clock", "h"], sort=False):
        for model_name in MODELS:
            for slice_name, group, sliced in _slices(part):
                result = _probability_metrics(
                    sliced.target, sliced[model_name], sliced.identity)
                metrics.append({
                    "scenario": scenario, "clock": clock, "h": h,
                    "model": model_name, "slice": slice_name,
                    "group": group, **result,
                })
            edges = np.linspace(0.0, 1.0, 11)
            p = _clip(part[model_name])
            bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, 9)
            for bin_id in range(10):
                mask = bins == bin_id
                if mask.any():
                    reliability.append({
                        "scenario": scenario, "clock": clock, "h": h,
                        "model": model_name, "bin": bin_id,
                        "n": int(mask.sum()),
                        "predicted": float(p[mask].mean()),
                        "actual": float(part.target.to_numpy()[mask].mean()),
                    })
        bootstrap.extend(_bootstrap(part, scenario, clock, h))

    metrics = pd.DataFrame(metrics)
    bootstrap = pd.DataFrame(bootstrap)
    overall = metrics[
        metrics.slice.eq("ALL") & metrics.model.isin(
            ["identity", "hierarchical_beta"])
    ].pivot(index=["scenario", "clock", "h"], columns="model",
            values=["brier", "logloss", "ece"]).reset_index()
    overall.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple) else col for col in overall.columns
    ]
    ci = bootstrap.groupby(["scenario", "clock", "h"]).ci_high.max().rename(
        "max_brier_ci_high").reset_index()
    state = overall.merge(ci, on=["scenario", "clock", "h"])
    state["brier_delta"] = (
        state.brier_hierarchical_beta - state.brier_identity)
    state["logloss_delta"] = (
        state.logloss_hierarchical_beta - state.logloss_identity)
    state["ece_delta"] = state.ece_hierarchical_beta - state.ece_identity
    state["pass"] = (
        (state.brier_delta < 0.0)
        & (state.max_brier_ci_high < 0.0)
        & (state.logloss_delta < 0.0)
        & (state.ece_delta <= 0.01)
    )

    high_ece_rows = metrics[
        metrics.slice.eq("currency_year")
        & metrics.model.isin(["identity", "hierarchical_beta"])
    ].copy()
    high_ece_rows["high_ece"] = high_ece_rows.ece > .08
    high_ece = high_ece_rows.groupby("model").high_ece.sum().to_dict()
    source_files = [
        REGISTERED,
        Path("research/temperature_t20_hierarchical_calibration.py"),
        Path("research/temperature_t19_anytime_quality_audit.py"),
        Path("results/research/temperature/t17_spot_availability_repair/metadata.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
        Path("data/cbr_rates_2010_2026.json"),
    ]
    checks = {
        "fit_origin": FIT_ORIGIN.isoformat(),
        "fit_cutoff_ordinal": fit_cutoff,
        "all_training_labels_mature_before_cutoff": bool(
            (pd.DataFrame(fit_checks).latest_train_maturity_ord < fit_cutoff).all()),
        "all_sources_no_later_than_query": bool(
            (pd.to_datetime(predictions.source_at, utc=True)
             <= pd.to_datetime(predictions.query_at, utc=True)).all()),
        "opened_evaluation_only": True,
        "changes_push_policy": False,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
    }
    metadata = {
        "packet": "temperature-T20",
        "primary_candidate": "hierarchical_beta",
        "selection_on_open_period": False,
        "evaluation_rows": int(len(predictions)),
        "state_rows": int(len(state)),
        "passing_states": int(state["pass"].sum()),
        "high_ece_currency_year_rows": {
            key: int(value) for key, value in high_ece.items()},
        "checks": checks,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    return {
        "predictions": predictions,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "reliability": pd.DataFrame(reliability),
        "state": state,
        "coefficients": pd.DataFrame(coefficient_rows),
        "fit_checks": pd.DataFrame(fit_checks),
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(
        OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    result["reliability"].to_csv(OUT / "reliability_bins.csv", index=False)
    result["state"].to_csv(OUT / "state_gates.csv", index=False)
    result["coefficients"].to_csv(OUT / "coefficients.csv", index=False)
    result["fit_checks"].to_csv(OUT / "fit_checks.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    state = result["state"]
    print(json.dumps({
        "passing_states": int(state["pass"].sum()),
        "states": int(len(state)),
        "passes_by_h": state.groupby("h")["pass"].agg(
            ["sum", "count"]).to_dict("index"),
        "mean_deltas_by_h": state.groupby("h")[[
            "brier_delta", "logloss_delta", "ece_delta"
        ]].mean().to_dict("index"),
        "high_ece_currency_year_rows": result["metadata"][
            "high_ece_currency_year_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
