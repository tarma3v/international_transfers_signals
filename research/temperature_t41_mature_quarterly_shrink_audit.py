"""Reconstruction and causality audit for T41."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t41_mature_quarterly_shrink import (
    CANDIDATE,
    OUT,
    SOURCE,
    _load_input,
    _quarterly_shrink,
    run,
)


BASE_KEYS = {
    "publication_predictions": ["publication_date", "currency"],
    "states": ["quarter_origin"],
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
    original, _ = _quarterly_shrink(source)
    corrupt = source.copy()
    future = pd.to_datetime(corrupt.publication_date) > pd.Timestamp("2021-12-31")
    valid = future & corrupt.target.notna()
    corrupt.loc[valid, "target"] = 1.0 - corrupt.loc[valid, "target"]
    corrupt.loc[future, SOURCE] = 1.0 - corrupt.loc[future, SOURCE]
    corrupt.loc[future, "causal_prior"] = 1.0 - corrupt.loc[
        future, "causal_prior"]
    rebuilt, _ = _quarterly_shrink(corrupt)
    prefix = pd.to_datetime(original.publication_date) <= pd.Timestamp("2021-12-31")
    columns = [CANDIDATE, "quarter_alpha", "quarter_origin"]
    if not original.loc[prefix, columns].astype(str).reset_index(
            drop=True).equals(rebuilt.loc[prefix, columns].astype(str).reset_index(
                drop=True)):
        raise AssertionError("future corruption changed T41 historical prefix")
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

    states = rebuilt["states"]
    if states.quarter_origin.duplicated().any():
        raise AssertionError("quarter state duplicated")
    if not states.alpha.between(0.0, 1.0).all():
        raise AssertionError("quarter alpha outside unit interval")
    used = states.latest_feedback_maturity_ord.notna()
    if not (states.loc[used, "latest_feedback_maturity_ord"]
            < states.loc[used, "cutoff_ord"]).all():
        raise AssertionError("quarter feedback violates embargo")
    predictions = rebuilt["publication_predictions"]
    if predictions[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("publication prediction duplicated")
    future_prefix = _future_prefix_check()
    decision = bool(rebuilt["historical_gates"]["pass"].all())
    if decision != metadata["historical_gate_passed"]:
        raise AssertionError("historical decision mismatch")
    if metadata["open_evaluated"] != decision:
        raise AssertionError("open evaluation did not follow historical gate")
    if metadata["production_promoted"] is not False:
        raise AssertionError("T41 retrospective promoted")
    result = {
        "source_hashes_match": True,
        "quarter_states_rebuilt": True,
        "publication_predictions_rebuilt": True,
        "historical_outputs_rebuilt": True,
        "feedback_embargo_passed": True,
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
