"""Independent reconstruction audit for the T35 h20 phase router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t19_anytime_quality_audit import BLOCKS, CLOCKS, SCENARIOS
from research.temperature_t35_unified_h20_phase_router import (
    AFTER_RECEIPT_CLOCKS,
    CANDIDATE,
    EARLY_CLOCKS,
    MODELS,
    OUT,
    _apply_phase_route,
    _asof_t34,
    _build_route,
    _load_inputs,
)


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
    state = pd.read_csv(OUT / "state_gates.csv")
    pooled = pd.read_csv(OUT / "pooled_metrics.csv")
    pooled_intervals = pd.read_csv(OUT / "pooled_paired_bootstrap.csv")

    if metadata["candidate"] != CANDIDATE:
        raise AssertionError("candidate metadata mismatch")
    if tuple(metadata["early_clocks"]) != EARLY_CLOCKS:
        raise AssertionError("early clock contract mismatch")
    if tuple(metadata["after_receipt_replay_clocks"]) != AFTER_RECEIPT_CLOCKS:
        raise AssertionError("receipt clock contract mismatch")
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
    if not (pd.to_datetime(predictions.source_at, utc=True)
            <= pd.to_datetime(predictions.query_at, utc=True)).all():
        raise AssertionError("source timestamp after query")
    if not (pd.to_datetime(predictions.t34_publication_date).dt.date
            <= pd.to_datetime(predictions.query_date).dt.date).all():
        raise AssertionError("future T34 publication used")

    t22, t34 = _load_inputs()
    rebuilt = _build_route(t22, t34)
    rebuilt_key = rebuilt[key].astype(str)
    saved = predictions.sort_values(key).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(key).reset_index(drop=True)
    if not saved[key].astype(str).equals(rebuilt[key].astype(str)):
        raise AssertionError("saved keys do not match source reconstruction")
    if not np.allclose(saved[CANDIDATE], rebuilt[CANDIDATE], atol=1e-12):
        raise AssertionError("route prediction cannot be reconstructed")
    if not saved.route_source.equals(rebuilt.route_source):
        raise AssertionError("route source cannot be reconstructed")

    early = rebuilt.clock.isin(EARLY_CLOCKS)
    receipt = (
        rebuilt.scenario.eq("calendar_assumed_replay")
        & rebuilt.clock.isin(AFTER_RECEIPT_CLOCKS))
    market = ~(early | receipt)
    if not np.allclose(rebuilt.loc[early, CANDIDATE],
                       rebuilt.loc[early, "t34_probability"], atol=0.0):
        raise AssertionError("early route does not use T34")
    if not np.allclose(rebuilt.loc[receipt, CANDIDATE],
                       rebuilt.loc[receipt, "rank_correction_selected"], atol=0.0):
        raise AssertionError("receipt route does not use T22")
    if not np.allclose(rebuilt.loc[market, CANDIDATE],
                       rebuilt.loc[market, "identity_h20"], atol=0.0):
        raise AssertionError("market/hold route changed baseline")
    no_receipt = rebuilt.scenario.eq("no_same_day_receipt")
    if rebuilt.loc[no_receipt].route_source.eq("t22_after_receipt_replay").any():
        raise AssertionError("T22 used without receipt")

    joined = _asof_t34(t22, t34)
    corrupted_target = joined.copy()
    corrupted_target["target"] = 1.0 - corrupted_target.target
    corrupted_target["t34_target"] = 1.0 - corrupted_target.t34_target
    rerouted = _apply_phase_route(corrupted_target).sort_values(
        key).reset_index(drop=True)
    if not np.allclose(rebuilt[CANDIDATE], rerouted[CANDIDATE], atol=0.0):
        raise AssertionError("target corruption changes route")

    cutoff = pd.Timestamp("2025-07-01").date()
    future_t34 = t34.copy()
    future_t34.loc[future_t34.query_date >= cutoff,
                   "cold_identity_qstack_w125_r100"] = 0.999999
    future_route = _build_route(t22, future_t34).sort_values(
        key).reset_index(drop=True)
    prefix = rebuilt.query_date < cutoff
    if not np.allclose(
            rebuilt.loc[prefix, CANDIDATE],
            future_route.loc[prefix, CANDIDATE], atol=0.0):
        raise AssertionError("future T34 corruption changes historical prefix")

    expected_states = {(s, c) for s in SCENARIOS for c in CLOCKS}
    if set(zip(state.scenario, state.clock)) != expected_states:
        raise AssertionError("state gate grid incomplete")
    if metadata["state_rows"] != len(expected_states):
        raise AssertionError("state count metadata mismatch")
    expected_modified = len(EARLY_CLOCKS) * len(SCENARIOS) + len(AFTER_RECEIPT_CLOCKS)
    if int(state.modified.sum()) != expected_modified:
        raise AssertionError("modified state count mismatch")
    if metadata["modified_states"] != expected_modified:
        raise AssertionError("modified state metadata mismatch")

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

    retrospective_passed = bool(
        state.loc[state.modified, "pass"].all()
        and state.loc[~state.modified, "pass"].all())
    if retrospective_passed != metadata["retrospective_route_passed"]:
        raise AssertionError("retrospective route decision mismatch")

    result = {
        "source_hashes_match": True,
        "route_predictions_rebuilt": True,
        "t34_backward_join_causal": True,
        "t34_used_only_at_registered_early_clocks": True,
        "t22_used_only_in_after_receipt_replay": True,
        "no_receipt_never_uses_t22": True,
        "market_and_hold_states_unchanged": True,
        "target_corruption_invariant": True,
        "future_t34_prefix_corruption_invariant": True,
        "query_keys_unique_and_complete": True,
        "probabilities_finite_and_bounded": True,
        "state_metric_reliability_bootstrap_grids_complete": True,
        "pooled_metric_bootstrap_grids_complete": True,
        "retrospective_route_passed": retrospective_passed,
        "production_promoted": False,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
