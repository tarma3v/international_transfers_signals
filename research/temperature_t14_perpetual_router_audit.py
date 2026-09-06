"""Integrity, provenance and coverage audit for T14."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of
from research.temperature_t14_perpetual_router import (
    HORIZONS,
    MOSCOW,
    OUT,
    SELECTED,
    T12,
    T13,
    build_snapshots,
)


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    snapshots = pd.read_csv(OUT / "snapshots.csv.gz")
    snapshots.valid_from = pd.to_datetime(snapshots.valid_from, utc=True)
    snapshots.source_at = pd.to_datetime(snapshots.source_at, utc=True)
    rebuilt, rebuilt_early = build_snapshots()
    pd.testing.assert_frame_equal(
        snapshots.reset_index(drop=True), rebuilt.reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)
    assert len(snapshots) == metadata["snapshot_rows"]
    assert not snapshots.duplicated(["currency", "valid_from"]).any()
    assert (snapshots.source_at <= snapshots.valid_from).all()

    early = snapshots[snapshots.source_kind.eq("mixed_horizon_prefix")]
    assert len(early) == len(rebuilt_early) == metadata["new_0900_rows"]
    local = early.valid_from.dt.tz_convert(MOSCOW)
    assert (local.dt.hour.eq(9) & local.dt.minute.eq(0)).all()
    for h in HORIZONS:
        probability = snapshots[f"probability_h{h}"].to_numpy(float)
        expected = snapshots[f"expected_future_bps_h{h}"].to_numpy(float)
        assert np.all(np.isfinite(probability))
        assert np.all((probability >= 0.) & (probability <= 1.))
        assert np.all(np.isfinite(expected))
        source = pd.to_datetime(early[f"source_at_h{h}"], utc=True)
        assert (source <= early.valid_from).all()
        if h in (1, 3):
            assert early[f"source_kind_h{h}"].eq(
                "moex_perpetual_prefix").all()
            assert early[f"phase_h{h}"].eq("market_prefix").all()
            assert early[f"confidence_h{h}"].eq("mature_history").all()
        else:
            assert early[f"source_kind_h{h}"].eq("cbr_history").all()
            assert early[f"phase_h{h}"].eq("premarket").all()
            assert early[f"confidence_h{h}"].eq("limited").all()

    selection = pd.read_csv(T13 / "selection.csv").set_index(["clock", "h"])
    assert selection.loc[("perp_0900", 1), "selected"] == SELECTED[1]
    assert selection.loc[("perp_0900", 3), "selected"] == SELECTED[3]
    assert all(selection.loc[("perp_0900", h), "selected"] == "control"
               for h in (5, 10, 20))

    base = pd.read_csv(T12 / "snapshots.csv.gz")
    base.valid_from = pd.to_datetime(base.valid_from, utc=True)
    base.source_at = pd.to_datetime(base.source_at, utc=True)
    common = list(base.columns)
    retained = snapshots[~snapshots.source_kind.eq(
        "mixed_horizon_prefix")][common].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        retained, base.reset_index(drop=True), check_dtype=False,
        check_exact=False, rtol=1e-14, atol=1e-14)

    examples = json.loads((OUT / "query_examples.json").read_text())
    assert examples["h1_0915"]["source_kind"] == "moex_perpetual_prefix"
    assert examples["h3_0915"]["source_kind"] == "moex_perpetual_prefix"
    assert examples["h5_0915"]["source_kind"] == "cbr_history"
    assert examples["h5_0915"]["confidence"] == "limited"
    assert examples["weekend_h1"]["freshness"] == "stale"

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

    query = dt.datetime(2025, 1, 10, 9, 15, tzinfo=MOSCOW)
    original = score_snapshot_as_of(snapshots, "KZT", query, 1)
    future = snapshots.iloc[[-1]].copy()
    future["currency"] = "KZT"
    future["valid_from"] = pd.Timestamp(
        dt.datetime(2027, 1, 1, 12, tzinfo=MOSCOW)).tz_convert("UTC")
    future["source_at"] = future.valid_from
    future["source_at_h1"] = future.valid_from
    future["probability_h1"] = 1. - original["probability_now_best_h"]
    altered = pd.concat([snapshots, future], ignore_index=True)
    assert score_snapshot_as_of(altered, "KZT", query, 1) == original

    old_push = base[base.push_now.astype(bool)].groupby("currency").size()
    new_push = snapshots[snapshots.push_now.astype(bool)].groupby("currency").size()
    pd.testing.assert_series_equal(old_push, new_push)
    checks = {
        "source_hashes_verified": True,
        "snapshot_artifact_exactly_rebuilt": True,
        "snapshot_rows": int(len(snapshots)),
        "new_0900_rows": int(len(early)),
        "t12_rows_unchanged": True,
        "horizon_specific_source_at_not_after_valid_from": True,
        "h1_h3_use_perpetual_h5_h10_h20_use_history": True,
        "all_horizon_values_finite_and_bounded": True,
        "selection_matches_frozen_t13_screen": True,
        "every_calendar_day_and_currency_queryable": True,
        "future_snapshot_corruption_prefix_invariant": True,
        "weekend_stale_behavior_verified": True,
        "push_counts_unchanged": new_push.to_dict(),
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(
        checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
