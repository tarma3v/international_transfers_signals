"""T34: an identity safety gate before quarterly h20 feedback exists."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t19_anytime_quality_audit import (
    EMBARGO_DAYS,
    _probability_metrics,
)
from research.temperature_t21_h20_curve_head import _fast_auc
from research.temperature_t26_delayed_base_rate import _paired_intervals
from research.temperature_t33_quarterly_stacking import _quarter_origin


T33_OUT = Path("results/research/temperature/t33_quarterly_stacking")
OUT = Path("results/research/temperature/t34_cold_start_identity_gate")
REGISTERED = Path("research/temperature_t34_cold_start_identity_gate_registered.md")
SOURCE_MODEL = "qstack_w125_r100"
CANDIDATE = "cold_identity_qstack_w125_r100"
MIN_FEEDBACK_BATCHES = 20
SCREEN_START = dt.date(2023, 1, 1)
VALIDATION_START = dt.date(2024, 1, 1)
EVALUATION_START = dt.date(2025, 1, 1)
MODELS = (
    "identity_early", "t25_base", "t34_candidate", "t34_selected", CANDIDATE,
)


def _load_inputs():
    development = pd.read_csv(T33_OUT / "development_predictions.csv.gz")
    evaluation = pd.read_csv(T33_OUT / "predictions.csv.gz")
    development["t25_base"] = development.identity_early
    columns = [
        "query_date", "year", "currency", "target", "maturity_ord",
        "identity_early", "t25_base", SOURCE_MODEL,
    ]
    frame = pd.concat([development[columns], evaluation[columns]], ignore_index=True)
    frame["query_date"] = pd.to_datetime(frame.query_date).dt.date
    if frame[["query_date", "currency"]].duplicated().any():
        raise AssertionError("T33 publication keys duplicated")

    states = pd.read_csv(T33_OUT / "states.csv")
    states = states[states.candidate.eq(SOURCE_MODEL)].copy()
    states["quarter_origin"] = pd.to_datetime(states.quarter_origin).dt.date
    if states.quarter_origin.duplicated().any():
        raise AssertionError("T33 source state duplicated")
    return frame, states


def _apply_gate(frame, states):
    output = frame.copy()
    output["quarter_origin"] = [
        _quarter_origin(value) for value in output.query_date]
    state = states[[
        "quarter_origin", "cutoff_ord", "eligible_feedback_batches",
        "selected_feedback_batches", "latest_feedback_publication_date",
        "latest_feedback_maturity_ord",
    ]].copy()
    state["cold_start"] = (
        state.eligible_feedback_batches < MIN_FEEDBACK_BATCHES)
    output = output.merge(
        state, on="quarter_origin", how="left", validate="many_to_one")
    if output.cold_start.isna().any():
        raise AssertionError("quarterly source state missing")
    output[CANDIDATE] = np.where(
        output.cold_start,
        output.identity_early,
        output[SOURCE_MODEL],
    )
    return output


def _stage_metrics(frame, mask):
    result = _probability_metrics(
        frame.loc[mask, "target"], frame.loc[mask, CANDIDATE],
        frame.loc[mask, "identity_early"])
    result["model"] = CANDIDATE
    result["auc_delta"] = result["auc"] - _fast_auc(
        frame.loc[mask, "target"], frame.loc[mask, "identity_early"])
    result["ece_delta"] = result["ece"] - result["baseline_ece"]
    result["feasible"] = bool(
        result["auc_delta"] >= 0.02
        and result["brier_delta"] < 0.0
        and result["logloss_delta"] < 0.0
        and result["ece_delta"] <= 0.005)
    return result


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
    for model in MODELS:
        for slice_name, group, part in _slices(frame):
            identity = _probability_metrics(
                part.target, part[model], part.identity_early)
            t25 = _probability_metrics(part.target, part[model], part.t25_base)
            rows.append({
                "model": model,
                "slice": slice_name,
                "group": group,
                **identity,
                "auc_delta_identity": identity["auc"] - _fast_auc(
                    part.target, part.identity_early),
                "ece_delta_identity": identity["ece"] - identity["baseline_ece"],
                "brier_delta_t25": t25["brier_delta"],
                "logloss_delta_t25": t25["logloss_delta"],
                "auc_delta_t25": t25["auc"] - _fast_auc(
                    part.target, part.t25_base),
                "ece_delta_t25": t25["ece"] - t25["baseline_ece"],
            })
    return pd.DataFrame(rows)


def _stage_slice_metrics(frame, stages):
    rows = []
    for stage, mask in stages.items():
        stage_rows = _metrics(frame.loc[mask].copy())
        stage_rows.insert(0, "stage", stage)
        rows.append(stage_rows)
    return pd.concat(rows, ignore_index=True)


def run():
    frame, source_states = _load_inputs()
    frame = _apply_gate(frame, source_states)
    dates = frame.query_date.to_numpy()
    screen_cutoff = (
        VALIDATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    validation_cutoff = (
        EVALUATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    screen = (
        (dates >= SCREEN_START) & (dates < VALIDATION_START)
        & (frame.maturity_ord.to_numpy() < screen_cutoff))
    validation = (
        (dates >= VALIDATION_START) & (dates < EVALUATION_START)
        & (frame.maturity_ord.to_numpy() < validation_cutoff))
    evaluation = dates >= EVALUATION_START
    if min(screen.sum(), validation.sum(), evaluation.sum()) < 1000:
        raise AssertionError("chronological split unexpectedly small")

    screen_result = _stage_metrics(frame, screen)
    validation_result = _stage_metrics(frame, validation)
    historical_protocol_passed = bool(
        screen_result["feasible"] and validation_result["feasible"])
    frame["t34_candidate"] = frame[CANDIDATE]
    frame["t34_selected"] = np.where(
        historical_protocol_passed, frame[CANDIDATE], frame.identity_early)

    evaluated = frame[evaluation].copy()
    development = frame[~evaluation].copy()
    metrics = _metrics(evaluated)
    stage_metrics = _stage_slice_metrics(frame, {
        "screen": screen,
        "validation": validation,
        "open_diagnostic": evaluation,
    })
    intervals = pd.concat([
        _paired_intervals(evaluated, CANDIDATE, "identity_early", "identity"),
        _paired_intervals(evaluated, CANDIDATE, "t25_base", "t25"),
    ], ignore_index=True)

    state_log = source_states.copy()
    state_log["cold_start"] = (
        state_log.eligible_feedback_batches < MIN_FEEDBACK_BATCHES)
    state_log["gate_output"] = np.where(
        state_log.cold_start, "identity_early", SOURCE_MODEL)

    source_files = [
        REGISTERED,
        Path("research/temperature_t34_cold_start_identity_gate.py"),
        Path("research/temperature_t34_cold_start_identity_gate_audit.py"),
        T33_OUT / "metadata.json",
        T33_OUT / "development_predictions.csv.gz",
        T33_OUT / "predictions.csv.gz",
        T33_OUT / "states.csv",
    ]
    metadata = {
        "packet": "temperature-T34",
        "candidate": CANDIDATE,
        "source_model": SOURCE_MODEL,
        "min_feedback_batches": MIN_FEEDBACK_BATCHES,
        "screen_passed": bool(screen_result["feasible"]),
        "validation_passed": bool(validation_result["feasible"]),
        "historical_protocol_passed": historical_protocol_passed,
        "selected_model": (
            CANDIDATE if historical_protocol_passed else "identity_early"),
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "open_period_previously_inspected": True,
        "fresh_independent_holdout": False,
        "splits": {
            "screen": "2023",
            "validation": "2024",
            "evaluation": "2025-2026",
            "screen_rows": int(screen.sum()),
            "validation_rows": int(validation.sum()),
            "evaluation_rows": int(evaluation.sum()),
            "screen_dates": int(frame.loc[screen, "query_date"].nunique()),
            "validation_dates": int(frame.loc[validation, "query_date"].nunique()),
            "evaluation_dates": int(frame.loc[evaluation, "query_date"].nunique()),
            "screen_cutoff_ord": screen_cutoff,
            "validation_cutoff_ord": validation_cutoff,
            "latest_screen_maturity_ord": int(
                frame.loc[screen, "maturity_ord"].max()),
            "latest_validation_maturity_ord": int(
                frame.loc[validation, "maturity_ord"].max()),
        },
        "screen": screen_result,
        "validation": validation_result,
        "checks": {
            "single_candidate_no_grid": True,
            "threshold_inherited_from_t33": MIN_FEEDBACK_BATCHES == 20,
            "cold_gate_uses_only_feedback_availability": True,
            "screen_validation_disjoint": bool(not (screen & validation).any()),
            "screen_mature": bool(
                frame.loc[screen, "maturity_ord"].max() < screen_cutoff),
            "validation_mature": bool(
                frame.loc[validation, "maturity_ord"].max() < validation_cutoff),
            "changes_push_policy": False,
            "changes_runtime_router": False,
            "historical_receipts_certified": False,
            "bank_execution_validated": False,
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    output_columns = [
        "query_date", "year", "currency", "target", "maturity_ord",
        "quarter_origin", "cutoff_ord", "eligible_feedback_batches",
        "selected_feedback_batches", "cold_start", "identity_early",
        "t25_base", SOURCE_MODEL, CANDIDATE, "t34_candidate", "t34_selected",
    ]
    return {
        "development_predictions": development[output_columns],
        "predictions": evaluated[output_columns],
        "states": state_log,
        "screen_validation": pd.DataFrame([
            {"stage": "screen", **screen_result},
            {"stage": "validation", **validation_result},
        ]),
        "metrics": metrics,
        "stage_metrics": stage_metrics,
        "paired_bootstrap": intervals,
        "metadata": metadata,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    result["development_predictions"].to_csv(
        OUT / "development_predictions.csv.gz", index=False, compression="gzip")
    result["predictions"].to_csv(
        OUT / "predictions.csv.gz", index=False, compression="gzip")
    result["states"].to_csv(OUT / "states.csv", index=False)
    result["screen_validation"].to_csv(
        OUT / "screen_validation.csv", index=False)
    result["metrics"].to_csv(OUT / "metrics.csv", index=False)
    result["stage_metrics"].to_csv(OUT / "stage_metrics.csv", index=False)
    result["paired_bootstrap"].to_csv(
        OUT / "paired_bootstrap.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(
        result["metadata"], ensure_ascii=False, indent=2))
    overall = result["metrics"][result["metrics"].slice.eq("ALL")]
    print(json.dumps({
        "screen": result["metadata"]["screen"],
        "validation": result["metadata"]["validation"],
        "historical_protocol_passed": result["metadata"][
            "historical_protocol_passed"],
        "production_promoted": result["metadata"]["production_promoted"],
        "evaluation": overall.set_index("model")[[
            "auc", "average_precision", "brier", "logloss", "ece",
            "brier_delta", "logloss_delta", "auc_delta_identity",
            "brier_delta_t25", "logloss_delta_t25", "auc_delta_t25",
        ]].to_dict("index"),
        "checks": result["metadata"]["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
