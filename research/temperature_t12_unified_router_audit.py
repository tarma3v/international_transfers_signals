"""Integrity, causality and coverage audit for T12."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of
from research.temperature_t12_unified_router import HORIZONS, MOSCOW, OUT


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    snapshots = pd.read_csv(OUT / "snapshots.csv.gz")
    snapshots.valid_from = pd.to_datetime(snapshots.valid_from, utc=True)
    snapshots.source_at = pd.to_datetime(snapshots.source_at, utc=True)
    assert len(snapshots) == metadata["snapshot_rows"]
    assert not snapshots.duplicated(["currency", "valid_from"]).any()
    assert (snapshots.source_at <= snapshots.valid_from).all()
    for h in HORIZONS:
        probability = snapshots[f"probability_h{h}"].to_numpy(float)
        expected = snapshots[f"expected_future_bps_h{h}"].to_numpy(float)
        assert np.all(np.isfinite(probability))
        assert np.all((probability >= 0.) & (probability <= 1.))
        assert np.all(np.isfinite(expected))

    early = snapshots[snapshots.source_kind.eq("moex_early_prefix")]
    assert len(early) == metadata["early_1000_rows"] and len(early)
    early_local = early.valid_from.dt.tz_convert(MOSCOW)
    source_local = early.source_at.dt.tz_convert(MOSCOW)
    assert (early_local.dt.hour.eq(10) & early_local.dt.minute.eq(0)).all()
    assert (source_local < early_local).all()
    assert early.availability_evidence.eq(
        "completed_same_day_cny_candle_strict_before_1000").all()
    assert set(snapshots[snapshots.phase.isin(
        ["premarket", "market_prefix", "pre_receipt_bridge"])]
        .benefit_source.dropna().unique()) <= {
            "mature_prior", "adaptive_h1_h3_h5_h10_prior_h20"}

    examples = json.loads((OUT / "query_examples.json").read_text())
    assert examples["premarket"]["phase"] == "premarket"
    assert examples["early_market"]["source_kind"] == "moex_early_prefix"
    assert examples["intraday"]["phase"] == "market_prefix"
    assert examples["after_decision"]["phase"] == "after_new_cbr"
    assert examples["after_market_update"]["phase"] == (
        "after_new_cbr_market_update")
    assert examples["weekend"]["freshness"] == "stale"

    premarket = snapshots[snapshots.source_kind.eq("cbr_history")]
    first = max(premarket[premarket.currency.eq(currency)].valid_from.min()
                .tz_convert(MOSCOW).date()
                for currency in snapshots.currency.unique())
    last = snapshots.valid_from.max().tz_convert(MOSCOW).date()
    for currency in sorted(snapshots.currency.unique()):
        part = snapshots[snapshots.currency.eq(currency)]
        for day in pd.date_range(first, last, freq="D").date:
            query = dt.datetime.combine(day, dt.time(12), tzinfo=MOSCOW)
            assert score_snapshot_as_of(part, currency, query, 5) is not None

    query = dt.datetime(2025, 1, 6, 10, 15, tzinfo=MOSCOW)
    original = score_snapshot_as_of(snapshots, "KZT", query, 5)
    future = snapshots.iloc[[-1]].copy()
    future["currency"] = "KZT"
    future["valid_from"] = pd.Timestamp(
        dt.datetime(2027, 1, 1, 12, tzinfo=MOSCOW)).tz_convert("UTC")
    future["source_at"] = future.valid_from
    future["probability_h5"] = 1. - original["probability_now_best_h"]
    altered = pd.concat([snapshots, future], ignore_index=True)
    assert score_snapshot_as_of(altered, "KZT", query, 5) == original

    push = snapshots[snapshots.push_now.astype(bool)]
    assert set(push.source_kind) <= {"cbr_receipt"}
    checks = {
        "source_hashes_verified": True,
        "snapshot_rows": int(len(snapshots)),
        "early_1000_rows": int(len(early)),
        "unique_currency_event_rows": True,
        "source_at_never_after_valid_from": True,
        "all_horizon_values_finite_and_bounded": True,
        "early_market_physical_timing_verified": True,
        "stable_benefit_source_policy_verified": True,
        "every_calendar_day_and_currency_queryable": True,
        "future_snapshot_corruption_prefix_invariant": True,
        "weekend_stale_behavior_verified": True,
        "push_only_on_after_decision_snapshot": True,
        "push_counts_by_currency": push.groupby("currency").size().to_dict(),
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(
        checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
