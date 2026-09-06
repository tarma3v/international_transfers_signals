"""T23: monthly delayed h20 calibration using only matured prior outcomes."""
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


OUT = Path("results/research/temperature/t23_h20_delayed_online")
REGISTERED = Path("research/temperature_t23_h20_delayed_online_registered.md")
PRIMARY = "delayed_selected"
MODELS = ("identity_h20", PRIMARY, "delayed_anchor", "delayed_joint")
MAX_HISTORY_ROWS = 1000
MIN_FIT_ROWS = 150
MIN_SCREEN_ROWS = 50
MAPPER_C = 0.05
PRIORITY = {"identity": 0, "anchor_logit": 1, "joint_logit": 2}


def _fit_mapper(family, fit, prediction):
    if family == "identity":
        return prediction.identity_h20.to_numpy(dtype=float), None
    columns = ["frozen_logit"]
    if family == "joint_logit":
        columns.append("residual_z")
    model = LogisticRegression(
        C=MAPPER_C, penalty="l2", solver="lbfgs", max_iter=3000,
        random_state=20260906)
    model.fit(fit[columns], fit.target.to_numpy(dtype=int))
    values = model.predict_proba(prediction[columns])[:, 1]
    return _clip(values), {
        "columns": columns,
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
    }


def _last_complete_dates(frame):
    dates = np.sort(frame.query_date.unique())
    keep = dates[-(MAX_HISTORY_ROWS // len(CORRIDORS)):]
    return frame[frame.query_date.isin(keep)].copy()


def _monthly_fit(history, evaluation):
    history = _last_complete_dates(history).sort_values(
        ["query_date", "currency"]).copy()
    dates = np.sort(history.query_date.unique())
    split = int(np.floor(len(dates) * .75))
    if split <= 0 or split >= len(dates):
        split = 0
    fit_dates = dates[:split]
    screen_dates = dates[split:]
    fit = history[history.query_date.isin(fit_dates)]
    screen = history[history.query_date.isin(screen_dates)]
    sufficient = bool(
        len(fit) >= MIN_FIT_ROWS and len(screen) >= MIN_SCREEN_ROWS
        and fit.target.nunique() == 2 and screen.target.nunique() == 2
        and history.target.nunique() == 2
    )
    selection_rows = []
    if not sufficient:
        identity = evaluation.identity_h20.to_numpy(dtype=float)
        return {
            PRIMARY: identity.copy(), "delayed_anchor": identity.copy(),
            "delayed_joint": identity.copy(),
        }, {
            "status": "fallback_identity", "selected_family": "identity",
            "history_rows": int(len(history)), "fit_rows": int(len(fit)),
            "screen_rows": int(len(screen)), "selection_rows": selection_rows,
            "selected_model": None, "anchor_model": None, "joint_model": None,
        }

    identity_metrics = _probability_metrics(
        screen.target, screen.identity_h20, screen.identity_h20)
    identity_metrics["ece_delta"] = 0.0
    selection_rows.append({
        "family": "identity", "feasible": True, **identity_metrics,
    })
    for family in ("anchor_logit", "joint_logit"):
        screen_prediction, _ = _fit_mapper(family, fit, screen)
        metrics = _probability_metrics(
            screen.target, screen_prediction, screen.identity_h20)
        metrics["ece_delta"] = metrics["ece"] - metrics["baseline_ece"]
        metrics["family"] = family
        metrics["feasible"] = bool(
            metrics["brier_delta"] <= 0.0
            and metrics["logloss_delta"] <= 0.0
            and metrics["ece_delta"] <= 0.01
        )
        selection_rows.append(metrics)
    feasible = [row for row in selection_rows if row["feasible"]]
    selected = sorted(
        feasible, key=lambda row: (-row["auc"], PRIORITY[row["family"]])
    )[0]["family"]

    anchor, anchor_model = _fit_mapper("anchor_logit", history, evaluation)
    joint, joint_model = _fit_mapper("joint_logit", history, evaluation)
    family_outputs = {
        "identity": evaluation.identity_h20.to_numpy(dtype=float),
        "anchor_logit": anchor,
        "joint_logit": joint,
    }
    return {
        PRIMARY: family_outputs[selected].copy(),
        "delayed_anchor": anchor,
        "delayed_joint": joint,
    }, {
        "status": "fit", "selected_family": selected,
        "history_rows": int(len(history)), "fit_rows": int(len(fit)),
        "screen_rows": int(len(screen)), "selection_rows": selection_rows,
        "selected_model": (
            None if selected == "identity"
            else anchor_model if selected == "anchor_logit" else joint_model),
        "anchor_model": anchor_model, "joint_model": joint_model,
    }


def _rank_stream(base, stream):
    X_base, X_stream, _unused, names = _encode(base, stream, stream.iloc[:1])
    ranker = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs", max_iter=5000,
        random_state=20260906)
    ranker.fit(X_base, base.target.to_numpy(dtype=int))
    raw_base = ranker.decision_function(X_base)
    raw_stream = ranker.decision_function(X_stream)
    frozen_base = _logit(base.identity_h20.to_numpy(dtype=float))
    frozen_stream = _logit(stream.identity_h20.to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(base)), frozen_base])
    residual_intercept, residual_slope = np.linalg.lstsq(
        design, raw_base, rcond=None)[0]
    residual_base = raw_base - (
        residual_intercept + residual_slope * frozen_base)
    residual_mean = float(residual_base.mean())
    residual_scale = float(residual_base.std(ddof=0))
    if not np.isfinite(residual_scale) or residual_scale < 1e-8:
        residual_scale = 1.0
    output = stream.copy()
    output["frozen_logit"] = frozen_stream
    output["residual_z"] = (
        raw_stream - (residual_intercept + residual_slope * frozen_stream)
        - residual_mean
    ) / residual_scale
    details = {
        "features": names,
        "rank_intercept": float(ranker.intercept_[0]),
        "rank_coefficients": [float(value) for value in ranker.coef_[0]],
        "residual_intercept": float(residual_intercept),
        "residual_slope": float(residual_slope),
        "residual_mean": residual_mean,
        "residual_scale": residual_scale,
    }
    return output, details


def _predict_state(base, stream):
    stream, rank_details = _rank_stream(base, stream)
    stream_date = pd.to_datetime(stream.query_date)
    evaluation = stream[stream_date >= EVAL_ORIGIN.tz_localize(None)].copy()
    month_origins = pd.to_datetime(evaluation.query_date).dt.to_period(
        "M").dt.to_timestamp().drop_duplicates().sort_values()
    outputs = []
    monthly_logs = []
    for origin in month_origins:
        cutoff = (origin - pd.Timedelta(days=EMBARGO_DAYS)).date().toordinal()
        history = stream[
            (stream_date < origin) & (stream.maturity_ord < cutoff)
        ].copy()
        month = evaluation[
            pd.to_datetime(evaluation.query_date).dt.to_period("M")
            == origin.to_period("M")
        ].copy()
        predictions, details = _monthly_fit(history, month)
        saved = month.copy()
        for name, values in predictions.items():
            saved[name] = values
        saved["monthly_origin"] = origin
        saved["selected_family"] = details["selected_family"]
        outputs.append(saved)
        latest_maturity = int(history.maturity_ord.max()) if len(history) else -1
        monthly_logs.append({
            "monthly_origin": origin.isoformat(),
            "fit_cutoff_ord": cutoff,
            "latest_eligible_maturity_ord": latest_maturity,
            **details,
        })
    return pd.concat(outputs, ignore_index=True), monthly_logs, rank_details


def _paired_intervals(part, scenario, clock):
    ordered = part.sort_values(["query_date", "currency"])
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
            20260908 + scenario_i * 100000 + clock_i * 1000 + block)
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
    predictions = []
    logs = []
    rank_details = []
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
            stream = part[
                valid & (date >= BASE_ORIGIN.tz_localize(None))].copy()
            if base.empty or stream.empty:
                raise AssertionError("empty base or stream")
            if not (base.maturity_ord < base_cutoff).all():
                raise AssertionError("immature base label")
            predicted, monthly_logs, rank = _predict_state(base, stream)
            saved = predicted[[
                "scenario", "query_date", "year", "clock", "currency",
                "query_at", "source_at", "phase", "target", "maturity_ord",
                "p20", "identity_h20", PRIMARY, "delayed_anchor",
                "delayed_joint", "monthly_origin", "selected_family",
            ]].copy()
            predictions.append(saved)
            for row in monthly_logs:
                logs.append({"scenario": scenario, "clock": clock, **row})
            rank_details.append({"scenario": scenario, "clock": clock, **rank})

    predictions = pd.concat(predictions, ignore_index=True)
    monthly_logs = pd.DataFrame([{
        key: value for key, value in row.items()
        if key not in {"selection_rows", "selected_model", "anchor_model", "joint_model"}
    } for row in logs])
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
        bounds = intervals[intervals.metric.eq(metric)].groupby(
            ["scenario", "clock"]).agg(
                ci_low=("ci_low", "min"), ci_high=("ci_high", "max")
        ).add_prefix(f"{metric}_").reset_index()
        state = state.merge(bounds, on=["scenario", "clock"])
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
    detailed_logs = []
    for row in logs:
        for candidate in row["selection_rows"]:
            detailed_logs.append({
                "scenario": row["scenario"], "clock": row["clock"],
                "monthly_origin": row["monthly_origin"],
                "selected_family": row["selected_family"], **candidate,
            })
    selection_metrics = pd.DataFrame(detailed_logs)
    source_files = [
        REGISTERED,
        Path("research/temperature_t23_h20_delayed_online.py"),
        Path("research/temperature_t22_h20_rank_correction.py"),
        Path("research/temperature_t19_anytime_quality_audit.py"),
        Path("results/research/temperature/t17_spot_availability_repair/metadata.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
        Path("data/cbr_rates_2010_2026.json"),
    ]
    metadata = {
        "packet": "temperature-T23",
        "primary_candidate": PRIMARY,
        "selection_on_open_period": False,
        "monthly_update": True,
        "max_history_rows": MAX_HISTORY_ROWS,
        "evaluation_rows": int(len(predictions)),
        "state_rows": int(len(state)),
        "passing_states": int(state["pass"].sum()),
        "monthly_choice_counts": {
            str(key): int(value) for key, value in
            monthly_logs.selected_family.value_counts().to_dict().items()},
        "high_ece_currency_year_rows": {
            key: int(value) for key, value in high_ece.items()},
        "checks": {
            "all_base_labels_mature": True,
            "all_monthly_labels_mature_before_cutoff": bool((
                monthly_logs.latest_eligible_maturity_ord
                < monthly_logs.fit_cutoff_ord).all()),
            "chronological_disjoint_fit_screen": True,
            "base_only_rank_and_residualization": True,
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
        "monthly_logs": monthly_logs, "selection_metrics": selection_metrics,
        "model_details": {"rank": rank_details, "monthly": logs},
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
    result["monthly_logs"].to_csv(OUT / "monthly_logs.csv", index=False)
    result["selection_metrics"].to_csv(
        OUT / "selection_metrics.csv", index=False)
    (OUT / "model_details.json").write_text(json.dumps(
        result["model_details"], ensure_ascii=False, indent=2))
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    state = result["state_gates"]
    overall = result["metrics"][result["metrics"].slice.eq("ALL")]
    print(json.dumps({
        "passing_states": int(state["pass"].sum()),
        "states": int(len(state)),
        "monthly_choice_counts": result["metadata"]["monthly_choice_counts"],
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
