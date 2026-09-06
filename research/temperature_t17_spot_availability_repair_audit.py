"""Independent audit for the T17 observed-availability repair."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.temperature_t8b_unified_router import HORIZONS, MOSCOW
from research.temperature_t17_spot_availability_repair import (
    BASE,
    OBSERVED_EVIDENCE,
    OUT,
    SPOT_SOURCE_KINDS,
    load_base,
    repair_snapshots,
)


def _canonical(frame):
    frame = frame.copy()
    for column in frame.columns:
        if column == "source_at" or column.startswith("source_at_h") \
                or column.startswith("benefit_source_at_h"):
            frame[column] = pd.to_datetime(
                frame[column], utc=True, errors="coerce", format="mixed")
    return frame.where(pd.notna(frame), np.nan)


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    base = load_base()
    history, _digest = load_spot_1530_history()
    saved = pd.read_csv(OUT / "snapshots.csv.gz")
    saved.valid_from = pd.to_datetime(saved.valid_from, utc=True)
    saved.source_at = pd.to_datetime(saved.source_at, utc=True)
    rebuilt, details = repair_snapshots(base, history)
    pd.testing.assert_frame_equal(
        _canonical(saved).reset_index(drop=True),
        _canonical(rebuilt).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14,
    )
    assert len(base) == metadata["base_rows"]
    assert len(saved) == metadata["snapshot_rows"]
    assert len(details) == metadata["planned_spot_rows"]
    assert int(details.observed.sum()) == metadata["observed_spot_rows"]
    assert int((~details.observed).sum()) == metadata[
        "dropped_synthetic_spot_rows"]
    assert metadata["dropped_synthetic_spot_rows"] > 0

    kept_keys = saved[["currency", "valid_from"]]
    common = base.merge(kept_keys, on=["currency", "valid_from"], how="inner")
    compare = saved.merge(
        common[["currency", "valid_from"]],
        on=["currency", "valid_from"], how="inner",
    )
    common = common.sort_values(["currency", "valid_from"]).reset_index(drop=True)
    compare = compare.sort_values(["currency", "valid_from"]).reset_index(drop=True)
    stable = [
        "currency", "valid_from", "source_kind", "phase", "confidence",
        "benefit_source", "push_now",
        *(f"probability_h{h}" for h in HORIZONS),
        *(f"expected_future_bps_h{h}" for h in HORIZONS),
    ]
    pd.testing.assert_frame_equal(
        _canonical(common[stable]), _canonical(compare[stable]),
        check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14,
    )

    spot = saved[saved.source_kind.isin(SPOT_SOURCE_KINDS)].copy()
    assert spot.availability_evidence.eq(OBSERVED_EVIDENCE).all()
    assert (spot.source_at < spot.valid_from).all()
    local_valid = spot.valid_from.dt.tz_convert(MOSCOW)
    local_source = spot.source_at.dt.tz_convert(MOSCOW)
    assert np.array_equal(local_valid.dt.date, local_source.dt.date)

    dropped = details.loc[~details.observed]
    assert dropped.observed_source_at.isna().all()
    push_counts = saved.groupby("currency").push_now.sum().astype(int).to_dict()
    expected_push = {"AMD": 142, "KGS": 139, "KZT": 138,
                     "TJS": 139, "UZS": 137}
    assert push_counts == expected_push

    weekend = dropped[pd.to_datetime(dropped.local_date).dt.dayofweek.ge(5)]
    assert len(weekend)
    example = weekend.iloc[0]
    query = pd.Timestamp(example.valid_from).tz_convert(MOSCOW).to_pydatetime()
    query += dt.timedelta(minutes=1)
    held = score_snapshot_as_of(saved, str(example.currency), query, 5)
    assert held is not None
    assert held["source_kind"] not in SPOT_SOURCE_KINDS
    assert pd.Timestamp(held["last_source_at"]) < pd.Timestamp(query)

    first = saved.valid_from.min().tz_convert(MOSCOW).date()
    last = saved.valid_from.max().tz_convert(MOSCOW).date()
    for day in pd.date_range(first, last, periods=24):
        query = dt.datetime.combine(day.date(), dt.time(12), tzinfo=MOSCOW)
        for currency in expected_push:
            if pd.Timestamp(query).tz_convert("UTC") >= saved[
                    saved.currency.eq(currency)].valid_from.min():
                assert score_snapshot_as_of(saved, currency, query, 5) is not None

    boundary = dt.datetime(2025, 1, 1)
    changed = {
        ticker: [row for row in rows if row["end"] < boundary]
        for ticker, rows in history.items()
    }
    altered, _details = repair_snapshots(base, changed)
    past_saved = saved[
        saved.valid_from.dt.tz_convert(MOSCOW).dt.date < boundary.date()
    ].reset_index(drop=True)
    past_altered = altered[
        altered.valid_from.dt.tz_convert(MOSCOW).dt.date < boundary.date()
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        _canonical(past_saved), _canonical(past_altered),
        check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14,
    )
    assert len(altered) < len(saved)

    checks = {
        "source_hashes_verified": True,
        "snapshot_artifact_exactly_rebuilt": True,
        "base_rows": int(len(base)),
        "snapshot_rows": int(len(saved)),
        "planned_spot_rows": int(len(details)),
        "observed_spot_rows": int(details.observed.sum()),
        "dropped_synthetic_spot_rows": int((~details.observed).sum()),
        "kept_predictions_benefits_push_exact": True,
        "observed_source_at_strictly_before_valid_from": True,
        "missing_weekend_uses_held_nonspot_source": True,
        "sampled_any_day_queries_available": True,
        "future_spot_deletion_prefix_invariant": True,
        "push_counts_unchanged": expected_push,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
