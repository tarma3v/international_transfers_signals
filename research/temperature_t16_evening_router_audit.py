"""Independent integrity, provenance and coverage audit for T16."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t8b_unified_router import HORIZONS, MOSCOW
from research.temperature_t16_evening_router import (
    AP,
    CLOCKS,
    OUT,
    T14,
    T15,
    build_snapshots,
)


def canonical_times(frame):
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

    snapshots = pd.read_csv(OUT / "snapshots.csv.gz")
    snapshots.valid_from = pd.to_datetime(snapshots.valid_from, utc=True)
    snapshots.source_at = pd.to_datetime(snapshots.source_at, utc=True)
    rebuilt, additions = build_snapshots()
    pd.testing.assert_frame_equal(
        canonical_times(snapshots).reset_index(drop=True),
        canonical_times(rebuilt).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)
    assert len(snapshots) == metadata["snapshot_rows"] == 60370
    assert len(additions) == metadata["new_evening_rows"] == 9855
    assert not snapshots.duplicated(["currency", "valid_from"]).any()
    assert (snapshots.source_at <= snapshots.valid_from).all()
    for h in HORIZONS:
        probability_source = pd.to_datetime(
            snapshots[f"source_at_h{h}"], utc=True, errors="coerce",
            format="mixed")
        benefit_source = pd.to_datetime(
            snapshots[f"benefit_source_at_h{h}"], utc=True, errors="coerce",
            format="mixed")
        valid_probability = probability_source.notna()
        valid_benefit = benefit_source.notna()
        assert (probability_source[valid_probability]
                <= snapshots.loc[valid_probability, "valid_from"]).all()
        assert (benefit_source[valid_benefit]
                <= snapshots.loc[valid_benefit, "valid_from"]).all()

    base = pd.read_csv(T14 / "snapshots.csv.gz")
    base.valid_from = pd.to_datetime(base.valid_from, utc=True)
    base.source_at = pd.to_datetime(base.source_at, utc=True)
    local = base.valid_from.dt.tz_convert(MOSCOW)
    non_2000 = ~(local.dt.hour.eq(20) & local.dt.minute.eq(0))
    retained = snapshots[
        ~snapshots.source_kind.eq("post_receipt_perpetual")]
    retained = retained[base.columns].sort_values(
        ["valid_from", "currency", "source_kind"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        canonical_times(retained[non_2000.to_numpy()]).reset_index(drop=True),
        canonical_times(base[non_2000]).reset_index(drop=True), check_dtype=False,
        check_exact=False, rtol=1e-14, atol=1e-14)
    allowed_2000 = {
        "expected_future_bps_h3",
        *(f"{field}_h{h}" for h in HORIZONS
          for field in ("source_at", "source_kind", "phase", "confidence",
                        "availability_evidence")),
    }
    stable_columns = [column for column in base.columns
                      if column not in allowed_2000]
    pd.testing.assert_frame_equal(
        canonical_times(retained.loc[
            ~non_2000.to_numpy(), stable_columns]).reset_index(drop=True),
        canonical_times(base.loc[
            ~non_2000, stable_columns]).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)

    panel = pd.read_csv(AP / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    t15 = loadz(T15 / "outputs.npz")
    selection = pd.read_csv(T15 / "selection.csv")
    assert not selection.query("kind == 'probability'").adopted.any()
    benefit_map = {
        "2000": {3}, "2100": {3, 5},
        "2200": {3, 5}, "2300": {3, 5},
    }
    probability_checks = benefit_checks = 0
    for i, item in panel.iterrows():
        if item.date.year < 2024:
            continue
        for clock in CLOCKS:
            valid = pd.Timestamp(
                dt.datetime(item.date.year, item.date.month, item.date.day,
                            int(clock[:2]), tzinfo=MOSCOW)).tz_convert("UTC")
            row = snapshots[
                snapshots.currency.eq(item.currency)
                & snapshots.valid_from.eq(valid)]
            assert len(row) == 1
            row = row.iloc[0]
            state = "perp_" + clock
            assert bool(row.push_now) is False
            np.testing.assert_allclose(
                float(row.probability_h1),
                float(t15["known_probability_h1"][i]),
                rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(
                float(row.expected_future_bps_h1),
                float(t15["known_future_bps_h1"][i]),
                rtol=1e-13, atol=1e-13)
            for h in (3, 5, 10, 20):
                np.testing.assert_allclose(
                    float(row[f"probability_h{h}"]),
                    float(t15[f"control_probability__{state}__h{h}"][i]),
                    rtol=1e-13, atol=1e-13)
                probability_checks += 1
                if h in benefit_map[clock]:
                    expected = t15[f"routed_benefit__{state}__h{h}"][i]
                    assert row[f"benefit_source_kind_h{h}"] == (
                        "post_receipt_perpetual")
                else:
                    expected = t15[f"control_benefit__{state}__h{h}"][i]
                    assert row[f"benefit_source_kind_h{h}"] == (
                        "post_receipt_market")
                np.testing.assert_allclose(
                    float(row[f"expected_future_bps_h{h}"]),
                    float(expected), rtol=1e-13, atol=1e-13)
                benefit_checks += 1

    example_day = panel.loc[
        pd.to_datetime(panel.date).dt.year.ge(2024), "date"].max()
    query = dt.datetime.combine(example_day, dt.time(23, 15), tzinfo=MOSCOW)
    h3 = score_snapshot_as_of(snapshots, "KZT", query, 3)
    h10 = score_snapshot_as_of(snapshots, "KZT", query, 10)
    assert h3["source_kind"] == "post_receipt_market"
    assert h3["benefit_source_kind"] == "post_receipt_perpetual"
    assert h3["last_source_at"].endswith("20:00:00+03:00")
    assert h3["benefit_last_source_at"].endswith("22:59:59+03:00")
    assert h10["source_kind"] == "post_receipt_market"
    assert h10["benefit_source_kind"] == "post_receipt_market"

    before = dt.datetime(2024, 12, 31, 23, 15, tzinfo=MOSCOW)
    original = score_snapshot_as_of(snapshots, "KZT", before, 5)
    corrupted = snapshots.copy()
    future = corrupted.valid_from >= pd.Timestamp(
        dt.datetime(2025, 1, 1, tzinfo=MOSCOW)).tz_convert("UTC")
    corrupted.loc[future, "probability_h5"] = .999999
    corrupted.loc[future, "expected_future_bps_h5"] = -999999.
    assert score_snapshot_as_of(corrupted, "KZT", before, 5) == original

    last_day = snapshots.valid_from.dt.tz_convert(MOSCOW).dt.date.max()
    weekend = last_day
    while weekend.weekday() < 5:
        weekend += dt.timedelta(days=1)
    stale = score_snapshot_as_of(
        snapshots, "KZT",
        dt.datetime.combine(weekend, dt.time(12), tzinfo=MOSCOW), 5)
    assert stale["freshness"] == "stale"
    assert stale["benefit_freshness"] == "stale"
    push_counts = snapshots.groupby("currency").push_now.sum().astype(int)
    expected_push = {"AMD": 142, "KGS": 139, "KZT": 138,
                     "TJS": 139, "UZS": 137}
    assert push_counts.to_dict() == expected_push

    checks = {
        "source_hashes_verified": True,
        "snapshot_artifact_exactly_rebuilt": True,
        "snapshot_rows": int(len(snapshots)),
        "new_evening_rows": int(len(additions)),
        "non_2000_t14_rows_unchanged": True,
        "only_expected_h3_and_provenance_changed_at_2000": True,
        "probability_values_carried_exactly": probability_checks,
        "benefit_values_routed_exactly": benefit_checks,
        "probability_and_benefit_provenance_separate": True,
        "all_sources_not_after_valid_from": True,
        "future_snapshot_corruption_prefix_invariant": True,
        "weekend_stale_behavior_verified": True,
        "push_counts_unchanged": expected_push,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
