"""Independent reconstruction audit for T38 local stability evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t19_anytime_quality_audit import BLOCKS, CLOCKS, SCENARIOS
from research.temperature_t38_h20_local_stability import (
    BOOTSTRAP_DRAWS,
    CANDIDATE,
    LOCAL_SLICES,
    OUT,
    _clock_local_metrics,
    _failure_summary,
    _load_predictions,
    _local_bootstrap,
    _pooled_gates,
    _pooled_local_metrics,
)


def _assert_frame_equal(saved, rebuilt, keys, name):
    saved = saved.sort_values(keys).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    if list(saved.columns) != list(rebuilt.columns):
        raise AssertionError(f"{name} columns mismatch")
    if not saved[keys].astype(str).equals(rebuilt[keys].astype(str)):
        raise AssertionError(f"{name} keys mismatch")
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


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = _load_predictions()
    rebuilt_clock = _clock_local_metrics()
    rebuilt_pooled = _pooled_local_metrics(predictions)
    rebuilt_intervals = _local_bootstrap(predictions)
    rebuilt_gates = _pooled_gates(rebuilt_pooled, rebuilt_intervals)
    rebuilt_summary = _failure_summary(rebuilt_clock, rebuilt_gates)

    saved_clock = pd.read_csv(OUT / "clock_local_metrics.csv")
    saved_pooled = pd.read_csv(OUT / "pooled_local_metrics.csv")
    saved_intervals = pd.read_csv(OUT / "local_paired_bootstrap.csv")
    saved_gates = pd.read_csv(OUT / "pooled_local_gates.csv")
    saved_summary = pd.read_csv(OUT / "failure_summary.csv")
    _assert_frame_equal(
        saved_clock, rebuilt_clock,
        ["scenario", "clock", "slice", "group"], "clock metrics")
    _assert_frame_equal(
        saved_pooled, rebuilt_pooled,
        ["scenario", "slice", "group"], "pooled metrics")
    _assert_frame_equal(
        saved_intervals, rebuilt_intervals,
        ["scenario", "slice", "group", "metric", "block_dates"],
        "local bootstrap")
    _assert_frame_equal(
        saved_gates, rebuilt_gates,
        ["scenario", "slice", "group"], "pooled gates")
    _assert_frame_equal(
        saved_summary, rebuilt_summary, ["slice"], "failure summary")

    currencies = set(predictions.currency.astype(str))
    years = set(predictions.year.astype(str))
    expected_clock = set()
    expected_pooled = set()
    for scenario in SCENARIOS:
        for clock in CLOCKS:
            expected_clock.update(
                (scenario, clock, "currency", currency)
                for currency in currencies)
            expected_clock.update(
                (scenario, clock, "year", year) for year in years)
            expected_clock.update(
                (scenario, clock, "currency_year", f"{currency}:{year}")
                for currency in currencies for year in years)
        expected_pooled.update(
            (scenario, "currency", currency) for currency in currencies)
        expected_pooled.update(
            (scenario, "year", year) for year in years)
        expected_pooled.update(
            (scenario, "currency_year", f"{currency}:{year}")
            for currency in currencies for year in years)
    if set(zip(saved_clock.scenario, saved_clock.clock,
               saved_clock.slice, saved_clock.group.astype(str))) != expected_clock:
        raise AssertionError("clock-local grid incomplete")
    if set(zip(saved_pooled.scenario, saved_pooled.slice,
               saved_pooled.group.astype(str))) != expected_pooled:
        raise AssertionError("pooled-local grid incomplete")
    expected_intervals = {
        (*key, metric, block)
        for key in expected_pooled
        for metric in ("brier", "logloss", "auc")
        for block in BLOCKS
    }
    if set(zip(saved_intervals.scenario, saved_intervals.slice,
               saved_intervals.group.astype(str), saved_intervals.metric,
               saved_intervals.block_dates)) != expected_intervals:
        raise AssertionError("local bootstrap grid incomplete")
    if not saved_intervals.draws.eq(BOOTSTRAP_DRAWS).all():
        raise AssertionError("bootstrap draw count mismatch")

    decision = bool(
        saved_clock.noninferior.all() and saved_gates["pass"].all())
    if decision != metadata["local_stability_passed"]:
        raise AssertionError("saved decision mismatch")
    if metadata["production_promoted"] is not False:
        raise AssertionError("diagnostic audit marked production")
    if metadata["model_changed"] is not False:
        raise AssertionError("diagnostic audit marked model mutation")
    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("diagnostic audit marked selection")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("open period mislabeled fresh")

    result = {
        "source_hashes_match": True,
        "t37_predictions_used_without_model_change": True,
        "clock_local_metrics_rebuilt": True,
        "pooled_local_metrics_rebuilt": True,
        "date_block_bootstrap_rebuilt": True,
        "expected_grids_complete": True,
        "bootstrap_draws_and_blocks_match": True,
        "decision_recomputed": True,
        "local_stability_passed": decision,
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
