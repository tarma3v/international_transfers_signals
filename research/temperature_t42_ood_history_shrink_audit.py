"""Reconstruction and causality audit for T42."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import temperature_t40_long_rolling_history as t40
from research.temperature_t42_ood_history_shrink import (
    CANDIDATE,
    ENERGY_FLOOR,
    OUT,
    SOURCE,
    _load_input,
    _ood_shrink,
    run,
)


BASE_KEYS = {
    "publication_predictions": ["publication_date", "currency"],
    "annual_ood_state": ["year"],
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
    source = _load_input()
    base = t40._load_base()
    original, _states, _ = _ood_shrink(source, base)
    boundary = pd.Timestamp("2021-12-31")
    corrupt_source = source.copy()
    future_source = pd.to_datetime(
        corrupt_source.publication_date) > boundary
    valid = future_source & corrupt_source.target.notna()
    corrupt_source.loc[valid, "target"] = 1.0 - corrupt_source.loc[
        valid, "target"]
    corrupt_source.loc[future_source, SOURCE] = 1.0 - corrupt_source.loc[
        future_source, SOURCE]
    corrupt_source.loc[future_source, "causal_prior"] = (
        1.0 - corrupt_source.loc[future_source, "causal_prior"])

    corrupt_base = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in base.items()
    }
    future_base = np.asarray([
        day > boundary.date() for day in corrupt_base["dates"]])
    corrupt_base["compact"][future_base] += 1000.0
    finite_target = future_base & np.isfinite(corrupt_base["target"])
    corrupt_base["target"][finite_target] = (
        1.0 - corrupt_base["target"][finite_target])
    rebuilt, _states, _ = _ood_shrink(corrupt_source, corrupt_base)
    prefix = pd.to_datetime(original.publication_date) <= boundary
    columns = [CANDIDATE, "ood_energy", "ood_alpha"]
    if not np.allclose(
            original.loc[prefix, columns].to_numpy(dtype=float),
            rebuilt.loc[prefix, columns].to_numpy(dtype=float),
            atol=1e-12, equal_nan=True):
        raise AssertionError("future corruption changed T42 historical prefix")
    return True


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"source hash mismatch: {name}")
    rebuilt = run()
    keys = dict(BASE_KEYS)
    if metadata["open_evaluated"]:
        keys.update(OPEN_KEYS)
    for name, frame_keys in keys.items():
        _assert_frame_equal(_saved_frame(name), rebuilt[name], frame_keys, name)

    states = rebuilt["annual_ood_state"]
    if states.year.duplicated().any() or set(states.year) != set(t40.YEARS):
        raise AssertionError("annual OOD state grid mismatch")
    if not (states.latest_training_maturity_ord < states.cutoff_ord).all():
        raise AssertionError("annual OOD state violates training embargo")
    predictions = rebuilt["publication_predictions"]
    if predictions[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("publication prediction duplicated")
    expected_alpha = np.minimum(
        1.0, 1.0 / np.maximum(
            predictions.ood_energy.to_numpy(dtype=float), ENERGY_FLOOR))
    if not np.allclose(
            predictions.ood_alpha, expected_alpha, atol=1e-12):
        raise AssertionError("OOD alpha formula mismatch")
    expected_probability = (
        predictions.causal_prior.to_numpy(dtype=float)
        + expected_alpha * (
            predictions[SOURCE].to_numpy(dtype=float)
            - predictions.causal_prior.to_numpy(dtype=float)))
    expected_probability = np.clip(
        expected_probability, t40.EPSILON, 1.0 - t40.EPSILON)
    if not np.allclose(
            predictions[CANDIDATE], expected_probability, atol=1e-12):
        raise AssertionError("OOD probability formula mismatch")
    future_prefix = _future_prefix_check()
    decision = bool(rebuilt["historical_gates"]["pass"].all())
    if decision != metadata["historical_gate_passed"]:
        raise AssertionError("historical decision mismatch")
    if metadata["open_evaluated"] != decision:
        raise AssertionError("open evaluation did not follow historical gate")
    if metadata["production_promoted"] is not False:
        raise AssertionError("T42 retrospective promoted")
    result = {
        "source_hashes_match": True,
        "annual_scaler_states_rebuilt": True,
        "publication_predictions_rebuilt": True,
        "historical_outputs_rebuilt": True,
        "training_embargo_passed": True,
        "alpha_formula_passed": True,
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
