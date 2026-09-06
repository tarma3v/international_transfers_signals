"""Independent reconstruction and causality audit for T43."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import temperature_t40_long_rolling_history as t40
from research.temperature_t43_nonlinear_history import (
    CANDIDATE,
    MODEL_PARAMS,
    OUT,
    _fit_year,
    run,
)


BASE_KEYS = {
    "publication_predictions": ["publication_date", "currency"],
    "annual_fit_log": ["year"],
    "historical_metrics": ["stage", "slice", "group"],
    "historical_paired_bootstrap": ["stage", "metric", "block_dates"],
    "historical_gates": ["stage"],
}
OPEN_KEYS = {
    "predictions": ["scenario", "query_date", "clock", "currency"],
    "metrics": ["scenario", "clock", "model", "slice", "group"],
    "reliability": ["scenario", "clock", "model", "bin_left", "bin_right"],
    "paired_bootstrap": ["scenario", "clock", "metric", "block_dates"],
    "state_summary": ["scenario", "clock"],
    "component_metrics": ["route_source"],
    "pooled_metrics": ["scenario", "model"],
    "pooled_paired_bootstrap": ["scenario", "metric", "block_dates"],
    "pooled_gate": ["scenario"],
    "pairwise_state_metrics": ["scenario", "clock"],
    "pairwise_pooled_metrics": ["scenario"],
    "clock_local_metrics": ["scenario", "clock", "slice", "group"],
    "pooled_local_metrics": ["scenario", "slice", "group"],
    "local_paired_bootstrap": [
        "scenario", "slice", "group", "metric", "block_dates"],
    "pooled_local_gates": ["scenario", "slice", "group"],
    "failure_summary": ["slice"],
}


def _saved_frame(name):
    suffix = ".csv.gz" if name in (
        "publication_predictions", "predictions") else ".csv"
    return pd.read_csv(OUT / f"{name}{suffix}")


def _assert_frame_equal(saved, rebuilt, keys, name):
    saved = saved.sort_values(keys).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    if list(saved.columns) != list(rebuilt.columns):
        raise AssertionError(f"{name} columns mismatch")
    numeric = saved.select_dtypes(include=[np.number]).columns
    if not np.allclose(
            saved[numeric].to_numpy(dtype=float),
            rebuilt[numeric].to_numpy(dtype=float),
            atol=1e-12, equal_nan=True):
        raise AssertionError(f"{name} numeric values mismatch")
    other = [column for column in saved.columns if column not in numeric]
    saved_other = saved[other].where(saved[other].notna(), "<MISSING>")
    rebuilt_other = rebuilt[other].where(
        rebuilt[other].notna(), "<MISSING>")
    if other and not saved_other.astype(str).equals(
            rebuilt_other.astype(str)):
        raise AssertionError(f"{name} values mismatch")


def _future_prefix_check():
    original = t40._load_base()
    original_frame, _ = _fit_year(original, 2021)
    corrupt = {
        key: (value.copy() if isinstance(value, np.ndarray) else list(value))
        for key, value in original.items()
    }
    future = corrupt["dates"] >= pd.Timestamp("2022-01-01").date()
    corrupt["compact"][future] = corrupt["compact"][future] * -7.0 + 123.0
    valid_future = future & np.isfinite(corrupt["target"])
    corrupt["target"][valid_future] = 1.0 - corrupt["target"][valid_future]
    corrupt["prior"], corrupt["prior_count"] = t40._causal_prior(
        corrupt["dates"], corrupt["target"], corrupt["maturity"])
    corrupt_frame, _ = _fit_year(corrupt, 2021)
    columns = [
        "raw_probability", "platt_probability", "causal_prior", CANDIDATE]
    if not np.array_equal(
            original_frame[columns].to_numpy(),
            corrupt_frame[columns].to_numpy()):
        raise AssertionError("future corruption changed T43 historical prefix")
    return True


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"source hash mismatch: {name}")
    if metadata["model_params"] != MODEL_PARAMS:
        raise AssertionError("T43 model parameters changed")

    rebuilt = run()
    keys = dict(BASE_KEYS)
    if metadata["open_evaluated"]:
        keys.update(OPEN_KEYS)
    for name, frame_keys in keys.items():
        _assert_frame_equal(_saved_frame(name), rebuilt[name], frame_keys, name)

    predictions = rebuilt["publication_predictions"]
    fit_log = rebuilt["annual_fit_log"]
    if set(predictions.year.unique()) != set(t40.YEARS):
        raise AssertionError("T43 annual year grid incomplete")
    if not predictions.groupby("year").currency.nunique().eq(5).all():
        raise AssertionError("T43 annual currency grid incomplete")
    if predictions[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("T43 publication key duplicated")
    if not fit_log.feature_count.eq(41).all():
        raise AssertionError("T43 feature count changed")
    if not fit_log.n_iter.eq(MODEL_PARAMS["max_iter"]).all():
        raise AssertionError("T43 iteration count changed")
    if not (fit_log.latest_training_maturity_ord
            < fit_log.fit_cutoff_ord).all():
        raise AssertionError("T43 training maturity violates embargo")
    if not (fit_log.latest_calibration_maturity_ord
            < fit_log.calibration_cutoff_ord).all():
        raise AssertionError("T43 calibration maturity violates embargo")
    if not predictions[CANDIDATE].between(0.0, 1.0).all():
        raise AssertionError("T43 probability outside unit interval")

    future_prefix = _future_prefix_check()
    decision = bool(rebuilt["historical_gates"]["pass"].all())
    if decision != metadata["historical_gate_passed"]:
        raise AssertionError("T43 historical decision mismatch")
    if metadata["open_evaluated"] != decision:
        raise AssertionError("T43 open evaluation did not follow gate")
    if metadata["production_promoted"] is not False:
        raise AssertionError("T43 retrospective promoted")
    result = {
        "source_hashes_match": True,
        "annual_fits_rebuilt": True,
        "publication_predictions_rebuilt": True,
        "historical_outputs_rebuilt": True,
        "fixed_model_parameters_passed": True,
        "maturity_embargo_passed": True,
        "future_prefix_corruption_passed": future_prefix,
        "historical_gate_passed": decision,
        "open_evaluated": metadata["open_evaluated"],
        "open_outputs_rebuilt": metadata["open_evaluated"],
        "production_promoted": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
