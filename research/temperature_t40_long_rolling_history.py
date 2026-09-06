"""T40: preregistered long annual rolling-origin h20 history expert.

The historical gate is evaluated on 2019-2024 before the already-open
2025-2026 anytime panel is touched.  A failed historical gate ends the
experiment and leaves T37 unchanged.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from ml.targets import build_targets
from research import temperature_t37_source_driven_h20_shrink50 as t37
from research import temperature_t38_h20_local_stability as t38
from research import temperature_t39_pre2025_currency_shrink as t39
from research.extended_features import CACHE as FEATURE_CACHE, LONG_DATA
from research.round5_features import load_round5_features
from research.temperature_t19_anytime_quality_audit import (
    BLOCKS,
    EMBARGO_DAYS,
    _probability_metrics,
)
from research.temperature_t21_h20_curve_head import _fast_auc, _logit
from research.temperature_t24_history_h20_anchor import COMPACT_FEATURES, _maturity


OUT = Path("results/research/temperature/t40_long_rolling_history")
REGISTERED = Path("research/temperature_t40_long_rolling_history_registered.md")
CANDIDATE = "long_annual_history_h20_shadow"
YEARS = tuple(range(2019, 2027))
SCREEN_YEARS = (2019, 2020, 2021, 2022)
VALIDATION_YEARS = (2023, 2024)
HISTORICAL_BOOTSTRAP_DRAWS = 1000
EPSILON = 1e-6
SEED = 20260906


def _clip(values):
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1.0 - EPSILON)


def _causal_prior(dates, target, maturity):
    """Global h20 rate using only labels mature before each query embargo."""
    valid = np.isfinite(target) & np.isfinite(maturity)
    event_maturity = maturity[valid].astype(int)
    event_values = target[valid].astype(float)
    order = np.argsort(event_maturity, kind="stable")
    event_maturity = event_maturity[order]
    event_values = event_values[order]
    cutoff = np.asarray(
        [day.toordinal() - EMBARGO_DAYS for day in dates], dtype=int)
    counts = np.searchsorted(event_maturity, cutoff, side="left")
    cumulative = np.cumsum(event_values)
    prior = np.full(len(dates), np.nan)
    has_history = counts > 0
    prior[has_history] = cumulative[counts[has_history] - 1] / counts[has_history]
    return prior, counts


def _load_base():
    X, names, index, series, *_ = load_round5_features()
    columns = [names.index(name) for name in COMPACT_FEATURES]
    compact = np.asarray(X[:, columns], dtype=float)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=str)
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index, h=20)
    prior, prior_count = _causal_prior(dates, target, maturity)
    finite_features = np.all(np.isfinite(compact), axis=1)
    if not finite_features.all():
        raise AssertionError("T40 compact feature matrix is not finite")
    return {
        "compact": compact,
        "dates": dates,
        "currencies": currencies,
        "target": target,
        "maturity": maturity,
        "prior": prior,
        "prior_count": prior_count,
        "feature_names": list(COMPACT_FEATURES),
    }


def _fit_year(base, year):
    dates = base["dates"]
    target = base["target"]
    maturity = base["maturity"]
    compact = base["compact"]
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
    if train.sum() < 1000 or calibration.sum() < 500 or query.sum() < 500:
        raise AssertionError(f"insufficient annual support for {year}")
    if np.unique(target[train]).size != 2 or np.unique(target[calibration]).size != 2:
        raise AssertionError(f"single-class annual fit for {year}")

    raw_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=SEED,
        ),
    )
    raw_model.fit(compact[train], target[train].astype(int))
    raw_calibration = raw_model.predict_proba(compact[calibration])[:, 1]
    raw_query = raw_model.predict_proba(compact[query])[:, 1]
    platt = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        random_state=SEED,
    )
    platt.fit(_logit(raw_calibration)[:, None], target[calibration].astype(int))
    calibrated = _clip(
        platt.predict_proba(_logit(raw_query)[:, None])[:, 1])
    candidate = t37._equal_logit_blend(base["prior"][query], calibrated)

    query_rows = np.flatnonzero(query)
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
        "source_row": query_rows,
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
        "raw_intercept": float(raw_model[-1].intercept_[0]),
        "raw_coefficients": json.dumps(
            [float(value) for value in raw_model[-1].coef_[0]]),
        "platt_intercept": float(platt.intercept_[0]),
        "platt_slope": float(platt.coef_[0, 0]),
    }
    return frame, log


def _annual_predictions(base=None):
    base = _load_base() if base is None else base
    frames, logs = [], []
    for year in YEARS:
        frame, log = _fit_year(base, year)
        frames.append(frame)
        logs.append(log)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("annual publication key is not unique")
    return predictions, pd.DataFrame(logs), base


def _metric_row(part):
    result = _probability_metrics(
        part.target, part[CANDIDATE], part.causal_prior)
    result["auc_delta"] = float(
        _fast_auc(part.target, part[CANDIDATE])
        - _fast_auc(part.target, part.causal_prior))
    result["ece_delta"] = float(result["ece"] - result["baseline_ece"])
    return result


def _historical_metrics(predictions):
    historical = predictions[
        predictions.year.isin(SCREEN_YEARS + VALIDATION_YEARS)
        & predictions.target.notna()
    ].copy()
    historical["stage"] = np.where(
        historical.year.isin(SCREEN_YEARS), "screen_2019_2022",
        "validation_2023_2024")
    rows = []
    for stage, part in historical.groupby("stage", sort=True):
        rows.append({"stage": stage, "slice": "pooled", "group": "all",
                     **_metric_row(part)})
        for year, group in part.groupby("year", sort=True):
            rows.append({"stage": stage, "slice": "year", "group": str(year),
                         **_metric_row(group)})
        for currency, group in part.groupby("currency", sort=True):
            rows.append({"stage": stage, "slice": "currency", "group": currency,
                         **_metric_row(group)})
    output = pd.DataFrame(rows)
    output["noninferior"] = (
        (output.brier_delta <= 0.001)
        & (output.logloss_delta <= 0.003)
        & (output.ece_delta <= 0.01)
        & (output.auc_delta >= -0.005)
    )
    return historical, output


def _historical_bootstrap_one(part, stage):
    ordered = part.sort_values(["publication_date", "currency"])
    dates = ordered.publication_date.drop_duplicates().to_numpy()
    counts = ordered.groupby("publication_date", sort=True).size().to_numpy()
    if len(set(counts.tolist())) != 1 or int(counts[0]) != len(CORRIDORS):
        raise AssertionError(f"{stage} is not a rectangular five-currency grid")
    width = len(CORRIDORS)
    y = ordered.target.to_numpy(dtype=int).reshape(len(dates), width)
    prior = _clip(ordered.causal_prior).reshape(len(dates), width)
    candidate = _clip(ordered[CANDIDATE]).reshape(len(dates), width)
    daily_brier = np.mean((candidate - y) ** 2 - (prior - y) ** 2, axis=1)
    daily_logloss = np.mean(
        -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
        + y * np.log(prior) + (1 - y) * np.log(1 - prior), axis=1)
    rows = []
    seed_base = int.from_bytes(
        hashlib.sha256(stage.encode()).digest()[:4], "big")
    for block in BLOCKS:
        rng = np.random.default_rng(seed_base + block)
        n_blocks = int(np.ceil(len(dates) / block))
        starts = rng.integers(
            0, len(dates), size=(HISTORICAL_BOOTSTRAP_DRAWS, n_blocks))
        samples = (
            starts[:, :, None] + np.arange(block)[None, None, :]
        ) % len(dates)
        samples = samples.reshape(HISTORICAL_BOOTSTRAP_DRAWS, -1)[:, :len(dates)]
        brier = daily_brier[samples].mean(axis=1)
        logloss = daily_logloss[samples].mean(axis=1)
        auc = np.empty(HISTORICAL_BOOTSTRAP_DRAWS)
        for draw_i, selected in enumerate(samples):
            ys = y[selected].ravel()
            auc[draw_i] = (
                _fast_auc(ys, candidate[selected].ravel())
                - _fast_auc(ys, prior[selected].ravel()))
        for metric, values in (("brier", brier), ("logloss", logloss), ("auc", auc)):
            rows.append({
                "stage": stage,
                "metric": metric,
                "block_dates": block,
                "draws": HISTORICAL_BOOTSTRAP_DRAWS,
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            })
    return rows


def _historical_bootstrap(historical):
    rows = []
    for stage, part in historical.groupby("stage", sort=True):
        rows.extend(_historical_bootstrap_one(part, stage))
    return pd.DataFrame(rows)


def _historical_gates(metrics, intervals):
    rows = []
    for stage in ("screen_2019_2022", "validation_2023_2024"):
        pooled = metrics[
            metrics.stage.eq(stage) & metrics.slice.eq("pooled")].iloc[0]
        local = metrics[
            metrics.stage.eq(stage) & metrics.slice.isin(["year", "currency"])]
        stage_intervals = intervals[intervals.stage.eq(stage)]
        brier_high = float(stage_intervals.loc[
            stage_intervals.metric.eq("brier"), "ci_high"].max())
        auc_low = float(stage_intervals.loc[
            stage_intervals.metric.eq("auc"), "ci_low"].min())
        pooled_pass = bool(
            pooled.brier_delta < 0.0
            and pooled.logloss_delta < 0.0
            and pooled.auc_delta > 0.0
            and pooled.ece_delta <= 0.01
            and brier_high < 0.0
            and auc_low > 0.0)
        local_pass = bool(local.noninferior.all())
        rows.append({
            "stage": stage,
            "pooled_pass": pooled_pass,
            "local_pass": local_pass,
            "pass": bool(pooled_pass and local_pass),
            "local_groups": int(len(local)),
            "local_noninferior": int(local.noninferior.sum()),
            "brier_delta": float(pooled.brier_delta),
            "logloss_delta": float(pooled.logloss_delta),
            "ece_delta": float(pooled.ece_delta),
            "auc_delta": float(pooled.auc_delta),
            "worst_brier_ci_high": brier_high,
            "worst_auc_ci_low": auc_low,
        })
    return pd.DataFrame(rows)


def _rename_t39(frame):
    output = frame.copy()
    output.columns = [
        str(column).replace(t39.CANDIDATE, CANDIDATE)
        for column in output.columns]
    if "model" in output:
        output["model"] = output.model.replace({t39.CANDIDATE: CANDIDATE})
    return output


def _open_outputs(publication_predictions):
    predictions = pd.read_csv(t37.OUT / "predictions.csv.gz")
    predictions["query_date"] = pd.to_datetime(predictions.query_date).dt.date
    predictions["t34_publication_date"] = pd.to_datetime(
        predictions.t34_publication_date).dt.date
    publication = publication_predictions[[
        "publication_date", "currency", CANDIDATE]].copy()
    publication["publication_date"] = pd.to_datetime(
        publication.publication_date).dt.date
    predictions = predictions.merge(
        publication,
        left_on=["t34_publication_date", "currency"],
        right_on=["publication_date", "currency"],
        how="left",
        validate="many_to_one",
    )
    history = predictions.snapshot_source_kind.eq("cbr_history")
    if predictions.loc[history, CANDIDATE].isna().any():
        raise AssertionError("T40 publication probability missing for history row")
    predictions.loc[~history, CANDIDATE] = predictions.loc[
        ~history, t37.CANDIDATE]
    predictions["modified_from_t37"] = (
        predictions[CANDIDATE].to_numpy()
        != predictions[t37.CANDIDATE].to_numpy())
    predictions["t40_route_source"] = predictions.route_source
    predictions.loc[history, "t40_route_source"] = "long_annual_cbr_history"
    predictions = predictions.drop(columns=["publication_date"])

    if not (pd.to_datetime(predictions.snapshot_source_at, utc=True)
            <= pd.to_datetime(predictions.query_at, utc=True)).all():
        raise AssertionError("source timestamp after query")
    non_history_exact = bool(np.array_equal(
        predictions.loc[~history, CANDIDATE].to_numpy(),
        predictions.loc[~history, t37.CANDIDATE].to_numpy()))
    predictions["year"] = pd.to_datetime(
        predictions.query_date).dt.year.astype(str)
    predictions["currency_year"] = (
        predictions.currency.astype(str) + ":" + predictions.year)

    work = predictions.copy()
    work[t39.CANDIDATE] = work[CANDIDATE]
    identity = {
        key: _rename_t39(value) if isinstance(value, pd.DataFrame) else value
        for key, value in t39._identity_outputs(work).items()
    }
    raw_identity = t39._identity_outputs(work)
    clock_local = _rename_t39(t39._clock_local_metrics(raw_identity["metrics"]))
    pooled_local = t39._pooled_local_metrics(work)
    local_intervals = t39._local_bootstrap(work)
    pooled_local_gates = t39._pooled_local_gates(
        pooled_local, local_intervals)
    failure_summary = t39._failure_summary(clock_local, pooled_local_gates)
    pairwise_state = t39._pairwise_metrics(work, ["scenario", "clock"])
    pairwise_pooled = t39._pairwise_metrics(work, ["scenario"])

    clock_passes = int(clock_local.noninferior.sum())
    pooled_passes = int(pooled_local_gates["pass"].sum())
    year_pass = bool(pooled_local_gates.loc[
        pooled_local_gates.slice.eq("year"), "pass"].all())
    passed = bool(
        pairwise_state.noninferior.all()
        and pairwise_pooled.noninferior.all()
        and year_pass
        and clock_passes > 619
        and pooled_passes > 25
        and non_history_exact)
    return {
        "predictions": predictions,
        **identity,
        "pairwise_state_metrics": pairwise_state,
        "pairwise_pooled_metrics": pairwise_pooled,
        "clock_local_metrics": clock_local,
        "pooled_local_metrics": pooled_local,
        "local_paired_bootstrap": local_intervals,
        "pooled_local_gates": pooled_local_gates,
        "failure_summary": failure_summary,
        "open_repair_passed": passed,
        "non_history_exact": non_history_exact,
    }


def _source_hashes():
    source_files = [
        REGISTERED,
        Path("research/temperature_t40_long_rolling_history.py"),
        Path("research/temperature_t40_long_rolling_history_audit.py"),
        LONG_DATA,
        FEATURE_CACHE,
        t37.OUT / "metadata.json",
        t37.OUT / "predictions.csv.gz",
        t38.OUT / "metadata.json",
        t38.OUT / "clock_local_metrics.csv",
        t38.OUT / "pooled_local_gates.csv",
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }


def run():
    publication_predictions, annual_fit_log, _ = _annual_predictions()
    historical, historical_metrics = _historical_metrics(publication_predictions)
    historical_intervals = _historical_bootstrap(historical)
    historical_gates = _historical_gates(
        historical_metrics, historical_intervals)
    historical_gate_passed = bool(historical_gates["pass"].all())
    open_outputs = (
        _open_outputs(publication_predictions)
        if historical_gate_passed else None)

    checks = {
        "annual_year_grid_complete": bool(
            set(publication_predictions.year.unique()) == set(YEARS)),
        "annual_currency_grid_complete": bool(
            set(publication_predictions.currency.unique()) == set(CORRIDORS)),
        "publication_key_unique": bool(not publication_predictions[[
            "publication_date", "currency"]].duplicated().any()),
        "probability_bounded": bool(publication_predictions[CANDIDATE].between(
            0.0, 1.0).all()),
        "fit_embargo_respected": bool((
            annual_fit_log.latest_training_maturity_ord
            < annual_fit_log.fit_cutoff_ord).all()),
        "calibration_embargo_respected": bool((
            annual_fit_log.latest_calibration_maturity_ord
            < annual_fit_log.calibration_cutoff_ord).all()),
        "open_not_evaluated_if_historical_failed": bool(
            historical_gate_passed or open_outputs is None),
    }
    if open_outputs is not None:
        checks.update({
            "source_no_later_than_query": True,
            "non_history_exact_t37": open_outputs["non_history_exact"],
        })
    metadata = {
        "packet": "temperature-T40",
        "candidate": CANDIDATE,
        "historical_screen": "annual OOS 2019-2022",
        "historical_validation": "annual OOS 2023-2024",
        "open_diagnostic": "2025-2026 only after historical pass",
        "annual_years": list(YEARS),
        "compact_features": list(COMPACT_FEATURES),
        "historical_bootstrap_draws": HISTORICAL_BOOTSTRAP_DRAWS,
        "bootstrap_blocks": list(BLOCKS),
        "historical_rows": int(len(historical)),
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
        "publication_predictions": publication_predictions,
        "annual_fit_log": annual_fit_log,
        "historical_metrics": historical_metrics,
        "historical_paired_bootstrap": historical_intervals,
        "historical_gates": historical_gates,
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
    (OUT / "metadata.json").write_text(
        json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "historical_gates": result["historical_gates"].to_dict("records"),
        "historical_gate_passed": result["metadata"]["historical_gate_passed"],
        "open_evaluated": result["metadata"]["open_evaluated"],
        "open_repair_passed": result["metadata"]["open_repair_passed"],
        "production_promoted": False,
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
