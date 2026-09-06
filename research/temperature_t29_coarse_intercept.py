"""T29: month/quarter-held causal intercept corrections for T25 h20."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS, _probability_metrics
from research.temperature_t21_h20_curve_head import _fast_auc, _logit
from research.temperature_t26_delayed_base_rate import (
    GLOBAL_CLIP,
    GLOBAL_RIDGE,
    MIN_DATES,
    _build_query,
    _paired_intervals,
    _solve_intercept,
)
from research.temperature_t27_rolling_origin_history import _historical_feedback


OUT = Path("results/research/temperature/t29_coarse_intercept")
REGISTERED = Path("research/temperature_t29_coarse_intercept_registered.md")
SCREEN_ORIGIN = dt.date(2024, 7, 1)
VALIDATION_ORIGIN = dt.date(2024, 10, 1)
EVALUATION_ORIGIN = dt.date(2025, 1, 1)
SPECS = {
    "quarter_w30_b025": {"frequency": "quarter", "window": 30, "beta": 0.25},
    "quarter_w30_b050": {"frequency": "quarter", "window": 30, "beta": 0.50},
    "quarter_w30_b100": {"frequency": "quarter", "window": 30, "beta": 1.00},
    "quarter_w60_b050": {"frequency": "quarter", "window": 60, "beta": 0.50},
    "quarter_w60_b100": {"frequency": "quarter", "window": 60, "beta": 1.00},
    "month_w30_b025": {"frequency": "month", "window": 30, "beta": 0.25},
    "month_w30_b050": {"frequency": "month", "window": 30, "beta": 0.50},
    "month_w30_b100": {"frequency": "month", "window": 30, "beta": 1.00},
    "month_w60_b050": {"frequency": "month", "window": 60, "beta": 0.50},
    "month_w60_b100": {"frequency": "month", "window": 60, "beta": 1.00},
}
CANDIDATES = tuple(SPECS)
PRIORITY = {name: i for i, name in enumerate(CANDIDATES)}
MODELS = ("identity_early", "t25_base", "coarse_selected", *CANDIDATES)


def _period_start(day, frequency):
    if frequency == "month":
        return dt.date(day.year, day.month, 1)
    month = 3 * ((day.month - 1) // 3) + 1
    return dt.date(day.year, month, 1)


def _next_period(boundary, frequency):
    months = 1 if frequency == "month" else 3
    month = boundary.month + months
    year = boundary.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return dt.date(year, month, 1)


def _coarse_candidates(frame, base, return_states=True):
    frame = frame.reset_index(drop=True)
    base = np.asarray(base, dtype=float)
    dates = pd.to_datetime(frame.query_date).dt.date.to_numpy()
    publication_dates = pd.to_datetime(frame.publication_date).dt.date.to_numpy()
    event = dates == publication_dates
    if frame.loc[event, ["publication_date", "currency"]].duplicated().any():
        raise AssertionError("publication feedback key duplicated")
    output = {name: base.copy() for name in CANDIDATES}
    states = []
    for name, spec in SPECS.items():
        frequency = spec["frequency"]
        boundaries = sorted({_period_start(day, frequency) for day in dates})
        for boundary in boundaries:
            end = _next_period(boundary, frequency)
            period = (dates >= boundary) & (dates < end)
            cutoff = (boundary - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
            eligible = (
                event
                & (publication_dates < boundary)
                & (frame.maturity_ord.to_numpy(dtype=float) < cutoff)
            )
            eligible_dates = np.asarray(sorted(
                frame.loc[eligible, "publication_date"].unique()))
            if len(eligible_dates) < MIN_DATES:
                chosen = np.asarray([], dtype=object)
                rows = np.zeros(len(frame), dtype=bool)
                raw_delta = 0.0
                latest_maturity = None
            else:
                chosen = eligible_dates[-spec["window"]:]
                rows = eligible & frame.publication_date.isin(chosen).to_numpy()
                raw_delta = _solve_intercept(
                    base[rows], frame.loc[rows, "target"],
                    GLOBAL_RIDGE, GLOBAL_CLIP)
                latest_maturity = int(frame.loc[rows, "maturity_ord"].max())
            applied_delta = float(spec["beta"] * raw_delta)
            output[name][period] = expit(_logit(base[period]) + applied_delta)
            if return_states:
                states.append({
                    "candidate": name,
                    "frequency": frequency,
                    "period_start": boundary,
                    "period_end": end,
                    "cutoff_ord": cutoff,
                    "feedback_dates": int(len(chosen)),
                    "feedback_rows": int(rows.sum()),
                    "latest_feedback_maturity_ord": latest_maturity,
                    "raw_delta": raw_delta,
                    "beta": spec["beta"],
                    "applied_delta": applied_delta,
                    "period_rows": int(period.sum()),
                })
    return output, pd.DataFrame(states), {
        "event_rows": int(event.sum()),
        "event_keys_unique": bool(not frame.loc[
            event, ["publication_date", "currency"]].duplicated().any()),
    }


def _candidate_metrics(frame, candidates, mask):
    base_auc = _fast_auc(frame.loc[mask, "target"], frame.loc[mask, "t25_base"])
    rows = []
    for name in CANDIDATES:
        result = _probability_metrics(
            frame.loc[mask, "target"], candidates[name][mask],
            frame.loc[mask, "t25_base"])
        result["ece_delta"] = result["ece"] - result["baseline_ece"]
        result["auc_delta"] = result["auc"] - base_auc
        result["model"] = name
        result["feasible"] = bool(
            result["brier_delta"] < 0.0
            and result["logloss_delta"] < 0.0
            and result["ece_delta"] <= 0.005
            and result["auc_delta"] >= -0.005)
        rows.append(result)
    return rows


def _choose(rows):
    feasible = [row for row in rows if row["feasible"]]
    return (
        sorted(feasible, key=lambda row: (row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "t25_base"
    )


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
    for name in MODELS:
        for slice_name, group, part in _slices(frame):
            result = _probability_metrics(part.target, part[name], part.t25_base)
            identity = _probability_metrics(part.target, part[name], part.identity_early)
            rows.append({
                "model": name, "slice": slice_name, "group": group, **result,
                "ece_delta_t25": result["ece"] - result["baseline_ece"],
                "auc_delta_t25": result["auc"] - _fast_auc(part.target, part.t25_base),
                "brier_delta_identity": identity["brier_delta"],
                "logloss_delta_identity": identity["logloss_delta"],
                "ece_delta_identity": identity["ece"] - identity["baseline_ece"],
                "auc_delta_identity": identity["auc"] - _fast_auc(part.target, part.identity_early),
            })
    return pd.DataFrame(rows)


def run():
    query, _query_selection, query_evaluation, model_details, split = _build_query()
    history, fit_log, history_details = _historical_feedback(
        model_details["t25_map_details"])
    query = query.copy()
    query["is_historical_feedback"] = False
    combined = pd.concat([
        history,
        query[[
            "query_date", "publication_date", "currency", "target",
            "maturity_ord", "identity_early", "t25_base", "year",
            "is_historical_feedback",
        ]],
    ], ignore_index=True, sort=False)
    n_history = len(history)
    candidates, states, feedback_details = _coarse_candidates(
        combined, combined.t25_base)

    dates = pd.to_datetime(combined.query_date).dt.date.to_numpy()
    maturity = combined.maturity_ord.to_numpy(dtype=float)
    is_query = ~combined.is_historical_feedback.to_numpy(dtype=bool)
    screen_cutoff = (
        VALIDATION_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    validation_cutoff = (
        EVALUATION_ORIGIN - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    screen = (
        is_query & (dates >= SCREEN_ORIGIN) & (dates < VALIDATION_ORIGIN)
        & (maturity < screen_cutoff))
    validation = (
        is_query & (dates >= VALIDATION_ORIGIN) & (dates < EVALUATION_ORIGIN)
        & (maturity < validation_cutoff))
    if screen.sum() < 150 or validation.sum() < 150:
        raise AssertionError("insufficient nested pre-2025 rows")

    screen_rows = _candidate_metrics(combined, candidates, screen)
    screen_selected = _choose(screen_rows)
    validation_rows = _candidate_metrics(combined, candidates, validation)
    validation_lookup = {row["model"]: row for row in validation_rows}
    validation_passed = bool(
        screen_selected != "t25_base"
        and validation_lookup[screen_selected]["feasible"])
    final_model = screen_selected if validation_passed else "t25_base"

    for name, values in candidates.items():
        query[name] = values[n_history:]
    query["coarse_selected"] = (
        query.t25_base if final_model == "t25_base" else query[final_model])
    evaluated = query[query_evaluation].copy()
    metrics = _metrics(evaluated)
    intervals = pd.concat([
        _paired_intervals(evaluated, "coarse_selected", "t25_base", "t25"),
        _paired_intervals(evaluated, "coarse_selected", "identity_early", "identity"),
    ], ignore_index=True)

    primary = metrics[(metrics.model == "coarse_selected") & (metrics.slice == "ALL")].iloc[0]
    year_rows = metrics[(metrics.model == "coarse_selected") & (metrics.slice == "year")]
    currency_rows = metrics[(metrics.model == "coarse_selected") & (metrics.slice == "currency")]
    brier_intervals = intervals[(intervals.comparison == "t25") & (intervals.metric == "brier")]
    passed = bool(
        final_model != "t25_base" and validation_passed
        and primary.brier_delta < 0.0 and primary.logloss_delta < 0.0
        and primary.ece_delta_t25 <= 0.005 and primary.auc_delta_t25 >= -0.005
        and (brier_intervals.ci_high < 0.0).all()
        and (year_rows.brier_delta <= 0.0).all()
        and (currency_rows.brier_delta <= 0.005).all())

    used_states = states[states.latest_feedback_maturity_ord.notna()]
    source_files = [
        REGISTERED,
        Path("research/temperature_t29_coarse_intercept.py"),
        Path("research/temperature_t28_weak_w30_blend.py"),
        Path("research/temperature_t27_rolling_origin_history.py"),
        Path("research/temperature_t26_delayed_base_rate.py"),
        Path("research/temperature_t25_anchor_preserving_map.py"),
        Path("results/research/temperature/t4_premarket/outputs.npz"),
        Path("research/cache/extended_features_2010_2026.npz"),
        Path("data/cbr_rates_2010_2026.json"),
        Path("results/research/temperature/t17_spot_availability_repair/snapshots.csv.gz"),
    ]
    metadata = {
        "packet": "temperature-T29",
        "screen_selected": screen_selected,
        "validation_passed": validation_passed,
        "selected_model": final_model,
        "selection_on_open_2025_2026": False,
        "pre2025_aggregate_previously_inspected": True,
        "fresh_independent_holdout": False,
        "passed": passed,
        "evaluation_rows": int(len(evaluated)),
        "evaluation_dates": int(evaluated.query_date.nunique()),
        "screen": screen_rows,
        "validation": validation_rows,
        "split": {
            **split,
            "nested_screen_rows": int(screen.sum()),
            "nested_validation_rows": int(validation.sum()),
            "nested_screen_cutoff_ord": screen_cutoff,
            "nested_validation_cutoff_ord": validation_cutoff,
            "latest_screen_maturity_ord": int(combined.loc[screen, "maturity_ord"].max()),
            "latest_validation_maturity_ord": int(combined.loc[validation, "maturity_ord"].max()),
        },
        "history_details": history_details,
        "feedback_details": feedback_details,
        "specs": SPECS,
        "checks": {
            "t25_selection_rebuilt": True,
            "historical_fit_causal": bool((
                fit_log.latest_training_maturity_ord < fit_log.cutoff_ord).all()),
            "screen_and_validation_disjoint": bool(not (screen & validation).any()),
            "screen_mature_at_validation_origin": bool(
                combined.loc[screen, "maturity_ord"].max() < screen_cutoff),
            "validation_mature_at_evaluation_origin": bool(
                combined.loc[validation, "maturity_ord"].max() < validation_cutoff),
            "period_feedback_causal": bool((
                used_states.latest_feedback_maturity_ord < used_states.cutoff_ord).all()),
            "period_delta_exact": bool(np.allclose(
                states.applied_delta, states.beta * states.raw_delta)),
            "publication_date_no_later_than_query": bool((
                pd.to_datetime(query.publication_date)
                <= pd.to_datetime(query.query_date)).all()),
            "all_horizon_sources_no_later_than_query": bool((
                pd.to_datetime(query.source_at, utc=True)
                <= pd.to_datetime(query.query_at, utc=True)).all()),
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
            "query_date", "year", "currency", "query_at", "source_at",
            "publication_date", "target", "maturity_ord", *MODELS]],
        "states": states,
        "historical_fit_log": fit_log,
        "metrics": metrics,
        "paired_bootstrap": intervals,
        "screen_validation": pd.DataFrame([
            {"stage": "screen", **row} for row in screen_rows
        ] + [{"stage": "validation", **row} for row in validation_rows]),
        "model_details": model_details,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["predictions"].to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["states"].to_csv(OUT / "states.csv.gz", index=False, compression="gzip")
    result["historical_fit_log"].to_csv(OUT / "historical_fit_log.csv", index=False)
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    result["screen_validation"].to_csv(OUT / "screen_validation.csv", index=False)
    (OUT / "model_details.json").write_text(json.dumps(
        result["model_details"], ensure_ascii=False, indent=2))
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    overall = result["metrics"][result["metrics"].slice == "ALL"]
    print(json.dumps({
        "screen_selected": result["metadata"]["screen_selected"],
        "validation_passed": result["metadata"]["validation_passed"],
        "selected_model": result["metadata"]["selected_model"],
        "passed": result["metadata"]["passed"],
        "screen": result["metadata"]["screen"],
        "validation": result["metadata"]["validation"],
        "evaluation": overall.set_index("model")[[
            "auc", "average_precision", "brier", "logloss", "ece",
            "brier_delta", "logloss_delta", "ece_delta_t25", "auc_delta_t25",
        ]].to_dict("index"),
        "paired_bootstrap": result["paired_bootstrap"].to_dict("records"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
