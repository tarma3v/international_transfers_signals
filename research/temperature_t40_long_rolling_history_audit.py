"""Independent reconstruction and causality audit for T40."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t40_long_rolling_history import (
    CANDIDATE,
    HISTORICAL_BOOTSTRAP_DRAWS,
    OUT,
    YEARS,
    _annual_predictions,
    _causal_prior,
    _fit_year,
    _load_base,
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
    suffix = ".csv.gz" if name in ("publication_predictions", "predictions") else ".csv"
    return pd.read_csv(OUT / f"{name}{suffix}")


def _assert_frame_equal(saved, rebuilt, keys, name):
    saved = saved.sort_values(keys).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    if list(saved.columns) != list(rebuilt.columns):
        raise AssertionError(f"{name} columns mismatch")
    for key in keys:
        if (pd.api.types.is_numeric_dtype(saved[key])
                and pd.api.types.is_numeric_dtype(rebuilt[key])):
            if not np.allclose(
                    saved[key].to_numpy(dtype=float),
                    rebuilt[key].to_numpy(dtype=float),
                    atol=1e-12, equal_nan=True):
                raise AssertionError(f"{name} key mismatch: {key}")
        elif not saved[key].astype(str).equals(rebuilt[key].astype(str)):
            raise AssertionError(f"{name} key mismatch: {key}")
    numeric = saved.select_dtypes(include=[np.number]).columns
    if not np.allclose(
            saved[numeric].to_numpy(dtype=float),
            rebuilt[numeric].to_numpy(dtype=float),
            atol=1e-12, equal_nan=True):
        raise AssertionError(f"{name} numeric values mismatch")
    other = [column for column in saved.columns
             if column not in numeric and column not in keys]
    if other and not saved[other].astype(str).equals(
            rebuilt[other].astype(str)):
        raise AssertionError(f"{name} values mismatch")


def _future_prefix_check():
    original = _load_base()
    original_frame, _ = _fit_year(original, 2021)
    corrupt = {
        key: (value.copy() if isinstance(value, np.ndarray) else list(value))
        for key, value in original.items()
    }
    future = corrupt["dates"] >= pd.Timestamp("2022-01-01").date()
    corrupt["compact"][future] = corrupt["compact"][future] * -7.0 + 123.0
    valid_future = future & np.isfinite(corrupt["target"])
    corrupt["target"][valid_future] = 1.0 - corrupt["target"][valid_future]
    corrupt["prior"], corrupt["prior_count"] = _causal_prior(
        corrupt["dates"], corrupt["target"], corrupt["maturity"])
    corrupt_frame, _ = _fit_year(corrupt, 2021)
    columns = [
        "raw_probability", "platt_probability", "causal_prior", CANDIDATE]
    if not np.array_equal(
            original_frame[columns].to_numpy(),
            corrupt_frame[columns].to_numpy()):
        raise AssertionError("future corruption changed the 2021 prefix")
    return True


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    rebuilt = run()
    keys = dict(BASE_KEYS)
    if metadata["open_evaluated"]:
        keys.update(OPEN_KEYS)
    for name, frame_keys in keys.items():
        _assert_frame_equal(_saved_frame(name), rebuilt[name], frame_keys, name)

    predictions = rebuilt["publication_predictions"]
    if set(predictions.year.unique()) != set(YEARS):
        raise AssertionError("annual year grid incomplete")
    annual_counts = predictions.groupby("year").currency.nunique()
    if not annual_counts.eq(5).all():
        raise AssertionError("annual currency grid incomplete")
    if predictions[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("annual publication key duplicated")
    if len(rebuilt["annual_fit_log"]) != len(YEARS):
        raise AssertionError("annual fit log incomplete")
    if not (rebuilt["annual_fit_log"].latest_training_maturity_ord
            < rebuilt["annual_fit_log"].fit_cutoff_ord).all():
        raise AssertionError("training maturity violates embargo")
    if not (rebuilt["annual_fit_log"].latest_calibration_maturity_ord
            < rebuilt["annual_fit_log"].calibration_cutoff_ord).all():
        raise AssertionError("calibration maturity violates embargo")
    intervals = rebuilt["historical_paired_bootstrap"]
    if len(intervals) != 2 * len((20, 50)) * 3:
        raise AssertionError("historical bootstrap grid incomplete")
    if not intervals.draws.eq(HISTORICAL_BOOTSTRAP_DRAWS).all():
        raise AssertionError("historical bootstrap draw count mismatch")

    future_prefix = _future_prefix_check()
    decision = bool(rebuilt["historical_gates"]["pass"].all())
    if decision != metadata["historical_gate_passed"]:
        raise AssertionError("historical decision mismatch")
    if metadata["open_evaluated"] != decision:
        raise AssertionError("open evaluation did not follow historical gate")
    if metadata["production_promoted"] is not False:
        raise AssertionError("open retrospective was promoted")

    result = {
        "source_hashes_match": True,
        "annual_fits_rebuilt": True,
        "causal_prior_rebuilt": True,
        "historical_metrics_rebuilt": True,
        "historical_bootstrap_rebuilt": True,
        "historical_gates_rebuilt": True,
        "annual_currency_grids_complete": True,
        "embargo_proofs_passed": True,
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
