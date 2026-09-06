"""Independent reconstruction audit for T39 pre-2025 currency shrink."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t39_pre2025_currency_shrink import (
    ALPHAS,
    BOOTSTRAP_DRAWS,
    CANDIDATE,
    OUT,
    T37_OUT,
    _development_split,
    _freeze_map,
    _open_predictions,
    run,
)


KEYS = {
    "screen_grid": ["currency", "alpha"],
    "validation_gates": ["currency"],
    "frozen_map": ["currency"],
    "predictions": ["scenario", "query_date", "clock", "currency"],
    "metrics": ["scenario", "clock", "model", "slice", "group"],
    "reliability": [
        "scenario", "clock", "model", "bin_left", "bin_right"],
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
                    atol=1e-12,
                    equal_nan=True):
                raise AssertionError(f"{name} key mismatch: {key}")
        elif not saved[key].astype(str).equals(rebuilt[key].astype(str)):
            raise AssertionError(f"{name} key mismatch: {key}")
    numeric = saved.select_dtypes(include=[np.number]).columns
    if not np.allclose(
        saved[numeric].to_numpy(dtype=float),
        rebuilt[numeric].to_numpy(dtype=float),
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError(f"{name} numeric values mismatch")
    other = [column for column in saved.columns
             if column not in numeric and column not in keys]
    if other and not saved[other].astype(str).equals(
            rebuilt[other].astype(str)):
        raise AssertionError(f"{name} values mismatch")


def _saved_frame(name):
    suffix = ".csv.gz" if name == "predictions" else ".csv"
    return pd.read_csv(OUT / f"{name}{suffix}")


def _future_prefix_check(frozen_map):
    source = pd.read_csv(T37_OUT / "predictions.csv.gz")
    original = _open_predictions(frozen_map, source)
    corrupt = source.copy()
    query_dates = pd.to_datetime(corrupt.query_date)
    cutoff = pd.Timestamp("2025-09-01")
    future = query_dates > cutoff
    corrupt.loc[future, "target"] = 1 - corrupt.loc[future, "target"]
    corrupt.loc[future, "identity_h20"] = (
        1 - corrupt.loc[future, "identity_h20"])
    corrupt.loc[future, "t34_probability"] = (
        1 - corrupt.loc[future, "t34_probability"])
    corrupt.loc[future, "source_driven_h20_shrink50_shadow"] = (
        1 - corrupt.loc[future, "source_driven_h20_shrink50_shadow"])
    rebuilt = _open_predictions(frozen_map, corrupt)
    prefix = pd.to_datetime(original.query_date) <= cutoff
    if not np.array_equal(
            original.loc[prefix, CANDIDATE].to_numpy(),
            rebuilt.loc[prefix, CANDIDATE].to_numpy()):
        raise AssertionError("future corruption changed historical prefix")
    return True


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    rebuilt = run()
    for name, keys in KEYS.items():
        _assert_frame_equal(
            _saved_frame(name), rebuilt[name], keys, name)

    development, screen, validation, _, _ = _development_split()
    grid, gates = _freeze_map(development, screen, validation)
    frozen_map = gates[[
        "currency", "selected_alpha", "validation_pass", "final_alpha",
        "changed_from_t37",
    ]].copy()
    if len(grid) != 5 * len(ALPHAS):
        raise AssertionError("screen alpha grid incomplete")
    if len(gates) != 5 or not gates.screen_rows.ge(200).all():
        raise AssertionError("currency selection support incomplete")
    if not gates.validation_rows.ge(200).all():
        raise AssertionError("currency validation support incomplete")
    if len(rebuilt["pairwise_state_metrics"]) != 40:
        raise AssertionError("pairwise clock grid incomplete")
    if len(rebuilt["pairwise_pooled_metrics"]) != 2:
        raise AssertionError("pairwise pooled grid incomplete")
    if len(rebuilt["clock_local_metrics"]) != 680:
        raise AssertionError("clock-local grid incomplete")
    if len(rebuilt["pooled_local_gates"]) != 34:
        raise AssertionError("pooled-local grid incomplete")
    if len(rebuilt["local_paired_bootstrap"]) != 204:
        raise AssertionError("local bootstrap grid incomplete")
    if not rebuilt["local_paired_bootstrap"].draws.eq(
            BOOTSTRAP_DRAWS).all():
        raise AssertionError("bootstrap draw count mismatch")

    predictions = rebuilt["predictions"]
    history = predictions.snapshot_source_kind.eq("cbr_history")
    if not np.array_equal(
            predictions.loc[~history, CANDIDATE].to_numpy(),
            predictions.loc[~history,
                            "source_driven_h20_shrink50_shadow"].to_numpy()):
        raise AssertionError("non-history T37 prediction changed")
    future_prefix = _future_prefix_check(frozen_map)

    decision = bool(
        rebuilt["metadata"]["retrospective_repair_passed"])
    if decision != metadata["retrospective_repair_passed"]:
        raise AssertionError("saved decision mismatch")
    if metadata["production_promoted"] is not False:
        raise AssertionError("open diagnostic marked production")
    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("open targets marked as selection data")

    result = {
        "source_hashes_match": True,
        "screen_grid_rebuilt": True,
        "validation_gates_rebuilt": True,
        "frozen_map_rebuilt": True,
        "open_predictions_rebuilt": True,
        "all_metric_packets_rebuilt": True,
        "expected_grids_complete": True,
        "pre2025_selection_only": True,
        "future_prefix_corruption_passed": future_prefix,
        "non_history_exact_t37": True,
        "retrospective_repair_passed": decision,
        "production_promoted": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
