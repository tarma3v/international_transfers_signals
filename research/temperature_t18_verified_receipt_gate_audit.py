"""Independent rebuild and timestamp audit for T18."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from ml.transfer_temperature import MOSCOW, receipt_gated_snapshots
from research.temperature_t18_verified_receipt_gate import OUT, run_audit


def _canonical(frame):
    frame = frame.copy().sort_values("corridor").reset_index(drop=True)
    frame = frame.astype(object)
    return frame.where(pd.notna(frame), "__NULL__")


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    rebuilt = run_audit()
    for name in (
        "without_receipt", "with_receipt", "late_receipt", "assumed_replay",
    ):
        saved = pd.read_csv(OUT / f"{name}.csv")
        pd.testing.assert_frame_equal(
            _canonical(saved), _canonical(rebuilt[name]),
            check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14,
        )

    for name, query in (
        ("without_receipt", dt.datetime.combine(
            rebuilt["day"], dt.time(18, 45), tzinfo=MOSCOW)),
        ("with_receipt", dt.datetime.combine(
            rebuilt["day"], dt.time(18, 45), tzinfo=MOSCOW)),
        ("late_receipt", dt.datetime.combine(
            rebuilt["day"], dt.time(19, 15), tzinfo=MOSCOW)),
    ):
        source = pd.to_datetime(rebuilt[name].last_source_at, utc=True)
        assert source.le(pd.Timestamp(query).tz_convert("UTC")).all()

    sample = pd.DataFrame({
        "currency": ["KZT"],
        "valid_from": ["2026-09-04T18:30:00+03:00"],
        "source_at": ["2026-09-04T18:30:00+03:00"],
        "source_kind": ["cbr_receipt"],
        "phase": ["after_new_cbr"],
        "probability_h5": [.5],
    })
    query = dt.datetime(2026, 9, 4, 18, 45, tzinfo=MOSCOW)
    for invalid in (
        "2026-09-04T19:00:00+03:00",
        "2026-09-03T18:30:00+03:00",
        "2026-09-04T18:30:00",
    ):
        try:
            receipt_gated_snapshots(
                sample, query, verified_receipt_at=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid receipt accepted: {invalid}")

    checks = {
        **rebuilt["checks"],
        "source_hashes_verified": True,
        "saved_outputs_exactly_rebuilt": True,
        "all_selected_sources_not_later_than_query": True,
        "invalid_receipt_events_rejected": True,
    }
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
