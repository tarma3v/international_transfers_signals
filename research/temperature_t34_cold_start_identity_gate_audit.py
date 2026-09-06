"""Independent reconstruction audit for the T34 cold-start identity gate."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t19_anytime_quality_audit import EMBARGO_DAYS
from research.temperature_t34_cold_start_identity_gate import (
    CANDIDATE,
    EVALUATION_START,
    MIN_FEEDBACK_BATCHES,
    MODELS,
    OUT,
    SCREEN_START,
    SOURCE_MODEL,
    VALIDATION_START,
    _apply_gate,
    _load_inputs,
    _stage_metrics,
)


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    development = pd.read_csv(OUT / "development_predictions.csv.gz")
    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    states = pd.read_csv(OUT / "states.csv")
    nested = pd.read_csv(OUT / "screen_validation.csv")
    metrics = pd.read_csv(OUT / "metrics.csv")
    stage_metrics = pd.read_csv(OUT / "stage_metrics.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")

    if metadata["candidate"] != CANDIDATE:
        raise AssertionError("candidate metadata mismatch")
    if metadata["source_model"] != SOURCE_MODEL:
        raise AssertionError("source model metadata mismatch")
    if metadata["min_feedback_batches"] != MIN_FEEDBACK_BATCHES:
        raise AssertionError("cold-start threshold mismatch")
    if metadata["production_promoted"] is not False:
        raise AssertionError("retrospective candidate marked as production")
    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("open period marked selector data")
    if metadata["open_period_previously_inspected"] is not True:
        raise AssertionError("opened-history disclosure missing")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("opened history mislabeled fresh")

    for frame in (development, predictions):
        if frame[["query_date", "currency"]].duplicated().any():
            raise AssertionError("publication key duplicated")
        probability = frame[list(MODELS) + [SOURCE_MODEL]]
        if not np.isfinite(probability.to_numpy()).all():
            raise AssertionError("non-finite probability")
        if not ((probability >= 0.0) & (probability <= 1.0)).all().all():
            raise AssertionError("probability outside [0,1]")
    combined = pd.concat([development, predictions], ignore_index=True)
    if combined[["query_date", "currency"]].duplicated().any():
        raise AssertionError("development/evaluation overlap")

    if states.quarter_origin.duplicated().any():
        raise AssertionError("duplicate quarterly gate state")
    expected_cold = states.eligible_feedback_batches < MIN_FEEDBACK_BATCHES
    if not np.array_equal(states.cold_start.to_numpy(), expected_cold.to_numpy()):
        raise AssertionError("cold-start gate mismatch")
    expected_output = np.where(expected_cold, "identity_early", SOURCE_MODEL)
    if not np.array_equal(states.gate_output.to_numpy(), expected_output):
        raise AssertionError("gate output state mismatch")
    used = states[~states.cold_start & states.latest_feedback_maturity_ord.notna()]
    if not (used.latest_feedback_maturity_ord < used.cutoff_ord).all():
        raise AssertionError("gate activated on immature feedback")
    if not (pd.to_datetime(used.latest_feedback_publication_date)
            < pd.to_datetime(used.quarter_origin)).all():
        raise AssertionError("gate activated on same/future quarter feedback")

    raw, source_states = _load_inputs()
    rebuilt = _apply_gate(raw, source_states)
    saved = combined.sort_values(["query_date", "currency"]).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(["query_date", "currency"]).reset_index(drop=True)
    if not saved[["query_date", "currency"]].astype(str).equals(
            rebuilt[["query_date", "currency"]].astype(str)):
        raise AssertionError("saved keys do not match T33 reconstruction")
    if not np.allclose(saved[CANDIDATE], rebuilt[CANDIDATE], atol=1e-12):
        raise AssertionError("candidate prediction cannot be reconstructed")
    expected = np.where(
        saved.cold_start, saved.identity_early, saved[SOURCE_MODEL])
    if not np.allclose(saved[CANDIDATE], expected, atol=1e-12):
        raise AssertionError("saved candidate violates gate")

    corrupted = raw.copy()
    corrupted["target"] = 1.0 - corrupted.target
    rebuilt_corrupted = _apply_gate(corrupted, source_states)
    if not np.allclose(rebuilt[CANDIDATE], rebuilt_corrupted[CANDIDATE], atol=0.0):
        raise AssertionError("future target changes gate prediction")

    dates = rebuilt.query_date.to_numpy()
    screen_cutoff = (
        VALIDATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    validation_cutoff = (
        EVALUATION_START - dt.timedelta(days=EMBARGO_DAYS)).toordinal()
    screen = (
        (dates >= SCREEN_START) & (dates < VALIDATION_START)
        & (rebuilt.maturity_ord.to_numpy() < screen_cutoff))
    validation = (
        (dates >= VALIDATION_START) & (dates < EVALUATION_START)
        & (rebuilt.maturity_ord.to_numpy() < validation_cutoff))
    rebuilt_screen = _stage_metrics(rebuilt, screen)
    rebuilt_validation = _stage_metrics(rebuilt, validation)
    saved_nested = nested.set_index("stage")
    for stage, row in (("screen", rebuilt_screen), ("validation", rebuilt_validation)):
        for field in ("auc", "brier", "logloss", "ece", "auc_delta"):
            if not np.isclose(saved_nested.loc[stage, field], row[field], atol=1e-12):
                raise AssertionError(f"{stage} {field} cannot be rebuilt")
        if bool(saved_nested.loc[stage, "feasible"]) != bool(row["feasible"]):
            raise AssertionError(f"{stage} gate cannot be rebuilt")
    historical_passed = bool(
        rebuilt_screen["feasible"] and rebuilt_validation["feasible"])
    if historical_passed != metadata["historical_protocol_passed"]:
        raise AssertionError("historical protocol decision mismatch")
    selected = CANDIDATE if historical_passed else "identity_early"
    if metadata["selected_model"] != selected:
        raise AssertionError("selected historical model mismatch")
    expected_selected = predictions[
        CANDIDATE if historical_passed else "identity_early"]
    if not np.allclose(predictions.t34_selected, expected_selected, atol=1e-12):
        raise AssertionError("selected predictions mismatch")

    if set(stage_metrics.stage) != {"screen", "validation", "open_diagnostic"}:
        raise AssertionError("stage metric periods incomplete")
    if set(stage_metrics.model) != set(MODELS):
        raise AssertionError("stage metric models incomplete")
    if len(metrics[metrics.slice.eq("ALL")]) != len(MODELS):
        raise AssertionError("open overall metric grid incomplete")
    expected_intervals = {
        (comparison, metric, block)
        for comparison in ("identity", "t25")
        for metric in ("auc", "brier", "logloss")
        for block in (20, 50)
    }
    if set(zip(intervals.comparison, intervals.metric,
               intervals.block_dates)) != expected_intervals:
        raise AssertionError("paired interval grid incomplete")

    result = {
        "source_hashes_match": True,
        "single_candidate_no_grid": True,
        "cold_gate_rebuilt_from_feedback_availability": True,
        "candidate_predictions_rebuilt": True,
        "future_target_corruption_invariant": True,
        "screen_passed": bool(rebuilt_screen["feasible"]),
        "validation_passed": bool(rebuilt_validation["feasible"]),
        "historical_protocol_passed": historical_passed,
        "production_promoted": False,
        "publication_keys_unique": True,
        "probabilities_finite_and_bounded": True,
        "stage_metric_grid_complete": True,
        "paired_interval_grid_complete": True,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
