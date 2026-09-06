"""Independent reconstruction audit for T36 source-driven h20 routing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import RECEIPT_DEPENDENT_SOURCE_KINDS
from research.temperature_t19_anytime_quality_audit import BLOCKS, CLOCKS, SCENARIOS
from research.temperature_t36_source_driven_h20_router import (
    CANDIDATE,
    MODELS,
    OUT,
    _apply_source_route,
    _build_route,
    _load_inputs,
)
from research.temperature_t35_unified_h20_phase_router import _asof_t34


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    metrics = pd.read_csv(OUT / "metrics.csv")
    reliability = pd.read_csv(OUT / "reliability_bins.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")
    state = pd.read_csv(OUT / "state_summary.csv")
    component = pd.read_csv(OUT / "component_metrics.csv")
    pooled = pd.read_csv(OUT / "pooled_metrics.csv")
    pooled_intervals = pd.read_csv(OUT / "pooled_paired_bootstrap.csv")
    pooled_gate = pd.read_csv(OUT / "pooled_gate.csv")

    if metadata["candidate"] != CANDIDATE:
        raise AssertionError("candidate metadata mismatch")
    if metadata["production_promoted"] is not False:
        raise AssertionError("opened route marked production")
    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("open period marked selector data")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("open data mislabeled fresh")
    if metadata["production_requires_verified_receipt_at"] is not True:
        raise AssertionError("receipt event requirement missing")

    key = ["scenario", "query_date", "clock", "currency"]
    if predictions[key].duplicated().any():
        raise AssertionError("saved query key duplicated")
    if set(predictions.scenario) != set(SCENARIOS):
        raise AssertionError("scenario grid incomplete")
    if set(predictions.clock) != set(CLOCKS):
        raise AssertionError("clock grid incomplete")
    if not np.isfinite(predictions[list(MODELS)].to_numpy()).all():
        raise AssertionError("non-finite probability")
    if not ((predictions[list(MODELS)] >= 0.0)
            & (predictions[list(MODELS)] <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")
    if not (pd.to_datetime(predictions.snapshot_source_at, utc=True)
            <= pd.to_datetime(predictions.query_at, utc=True)).all():
        raise AssertionError("source timestamp after query")
    if not (pd.to_datetime(predictions.t34_publication_date).dt.date
            <= pd.to_datetime(predictions.query_date).dt.date).all():
        raise AssertionError("future T34 publication used")

    t22, t34 = _load_inputs()
    rebuilt = _build_route(t22, t34).sort_values(key).reset_index(drop=True)
    saved = predictions.sort_values(key).reset_index(drop=True)
    if not saved[key].astype(str).equals(rebuilt[key].astype(str)):
        raise AssertionError("saved keys do not match reconstruction")
    if not np.allclose(saved[CANDIDATE], rebuilt[CANDIDATE], atol=1e-12):
        raise AssertionError("route prediction cannot be reconstructed")
    if not saved.route_source.equals(rebuilt.route_source):
        raise AssertionError("route source cannot be reconstructed")
    if not saved.snapshot_source_kind.equals(rebuilt.snapshot_source_kind):
        raise AssertionError("source kind cannot be reconstructed")

    history = rebuilt.snapshot_source_kind.eq("cbr_history")
    receipt = (
        rebuilt.scenario.eq("calendar_assumed_replay")
        & rebuilt.snapshot_source_kind.isin(RECEIPT_DEPENDENT_SOURCE_KINDS))
    unchanged = ~(history | receipt)
    if not rebuilt.loc[history].route_source.eq("t34_cbr_history").all():
        raise AssertionError("T34 not used on all and only CBR history rows")
    if not np.allclose(rebuilt.loc[history, CANDIDATE],
                       rebuilt.loc[history, "t34_probability"], atol=0.0):
        raise AssertionError("CBR history row does not use T34")
    if not rebuilt.loc[receipt].route_source.eq(
            "t22_after_receipt_replay").all():
        raise AssertionError("receipt row does not use T22")
    if not np.allclose(rebuilt.loc[receipt, CANDIDATE],
                       rebuilt.loc[receipt, "rank_correction_selected"], atol=0.0):
        raise AssertionError("receipt row prediction mismatch")
    if not np.array_equal(rebuilt.loc[unchanged, CANDIDATE].to_numpy(),
                          rebuilt.loc[unchanged, "identity_h20"].to_numpy()):
        raise AssertionError("unchanged source row differs from identity")
    no_receipt = rebuilt.scenario.eq("no_same_day_receipt")
    if rebuilt.loc[no_receipt].route_source.eq("t22_after_receipt_replay").any():
        raise AssertionError("T22 used without receipt")

    joined = _asof_t34(t22, t34)
    corrupted = joined.copy()
    corrupted["target"] = 1.0 - corrupted.target
    corrupted["t34_target"] = 1.0 - corrupted.t34_target
    rerouted = _apply_source_route(corrupted).sort_values(key).reset_index(drop=True)
    if not np.allclose(rebuilt[CANDIDATE], rerouted[CANDIDATE], atol=0.0):
        raise AssertionError("target corruption changes route")

    cutoff = pd.Timestamp("2025-07-01").date()
    future_t34 = t34.copy()
    future_t34.loc[future_t34.query_date >= cutoff,
                   "cold_identity_qstack_w125_r100"] = 0.999999
    future_route = _build_route(t22, future_t34).sort_values(key).reset_index(drop=True)
    prefix = rebuilt.query_date < cutoff
    if not np.allclose(rebuilt.loc[prefix, CANDIDATE],
                       future_route.loc[prefix, CANDIDATE], atol=0.0):
        raise AssertionError("future T34 corruption changes prefix")

    expected_states = {(s, c) for s in SCENARIOS for c in CLOCKS}
    if set(zip(state.scenario, state.clock)) != expected_states:
        raise AssertionError("state summary grid incomplete")
    overall = metrics[metrics.slice.eq("ALL")]
    if set(zip(overall.scenario, overall.clock, overall.model)) != {
            (s, c, m) for s in SCENARIOS for c in CLOCKS for m in MODELS}:
        raise AssertionError("overall metric grid incomplete")
    if set(zip(reliability.scenario, reliability.clock, reliability.model)) != {
            (s, c, m) for s in SCENARIOS for c in CLOCKS for m in MODELS}:
        raise AssertionError("reliability grid incomplete")
    expected_intervals = {
        (s, c, metric, block)
        for s in SCENARIOS for c in CLOCKS
        for metric in ("brier", "logloss", "auc") for block in BLOCKS
    }
    if set(zip(intervals.scenario, intervals.clock, intervals.metric,
               intervals.block_dates)) != expected_intervals:
        raise AssertionError("state bootstrap grid incomplete")
    if set(component.route_source) != {
            "t34_cbr_history", "t22_after_receipt_replay"}:
        raise AssertionError("component metric grid incomplete")
    if set(zip(pooled.scenario, pooled.model)) != {
            (s, m) for s in SCENARIOS for m in MODELS}:
        raise AssertionError("pooled metric grid incomplete")
    expected_pooled = {
        (s, metric, block) for s in SCENARIOS
        for metric in ("brier", "logloss", "auc") for block in BLOCKS
    }
    if set(zip(pooled_intervals.scenario, pooled_intervals.metric,
               pooled_intervals.block_dates)) != expected_pooled:
        raise AssertionError("pooled bootstrap grid incomplete")
    if set(pooled_gate.scenario) != set(SCENARIOS):
        raise AssertionError("pooled gate grid incomplete")

    retrospective = bool(
        pooled_gate["pass"].all()
        and component["pass"].all()
        and state.noninferior.all()
        and np.array_equal(rebuilt.loc[unchanged, CANDIDATE].to_numpy(),
                           rebuilt.loc[unchanged, "identity_h20"].to_numpy()))
    if retrospective != metadata["retrospective_route_passed"]:
        raise AssertionError("retrospective decision mismatch")

    result = {
        "source_hashes_match": True,
        "source_map_and_predictions_rebuilt": True,
        "source_timestamp_causal": True,
        "t34_backward_join_causal": True,
        "t34_used_only_on_cbr_history": True,
        "t22_used_only_on_receipt_replay": True,
        "no_receipt_never_uses_t22": True,
        "other_sources_bitwise_unchanged": True,
        "target_corruption_invariant": True,
        "future_t34_prefix_corruption_invariant": True,
        "query_keys_unique_and_complete": True,
        "metric_reliability_component_bootstrap_grids_complete": True,
        "retrospective_route_passed": retrospective,
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
