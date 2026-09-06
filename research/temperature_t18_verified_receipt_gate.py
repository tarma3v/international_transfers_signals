"""T18: production-safe receipt-event gate for the any-time router."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.transfer_temperature import (
    HORIZONS,
    MOSCOW,
    RECEIPT_DEPENDENT_SOURCE_KINDS,
    case_output_runtime_table_as_of,
    case_output_table_as_of,
    receipt_gated_snapshots,
)


OUT = Path("results/research/temperature/t18_verified_receipt_gate")
BASE = Path("results/research/temperature/t17_spot_availability_repair")


def load_base():
    frame = pd.read_csv(BASE / "snapshots.csv.gz")
    frame["valid_from"] = pd.to_datetime(frame.valid_from, utc=True)
    frame["source_at"] = pd.to_datetime(frame.source_at, utc=True)
    return frame


def _complete_example_day(frame):
    local = frame.valid_from.dt.tz_convert(MOSCOW)
    work = frame.assign(
        local_date=local.dt.date,
        clock=local.dt.strftime("%H:%M"),
    )
    required = {
        ("post_window_market", "17:30"),
        ("cbr_receipt", "18:30"),
        ("post_receipt_market", "19:00"),
    }
    candidate_days = None
    for source_kind, clock in required:
        part = work[
            work.source_kind.eq(source_kind) & work.clock.eq(clock)
        ]
        counts = part.groupby("local_date").currency.nunique()
        days = set(counts[counts.eq(len(CORRIDORS))].index)
        candidate_days = days if candidate_days is None else candidate_days & days
    if not candidate_days:
        raise AssertionError("no day has all receipt-gate example states")
    return max(candidate_days)


def _query(day, hour, minute):
    return dt.datetime.combine(day, dt.time(hour, minute), tzinfo=MOSCOW)


def _records(frame):
    columns = [
        "corridor", "source_kind", "phase", "probability_now_best_h",
        "expected_future_cbr_bps_h", "last_source_at", "receipt_verified",
        "receipt_at", "push_now",
    ]
    return frame[columns].sort_values("corridor").to_dict("records")


def run_audit():
    snapshots = load_base()
    day = _complete_example_day(snapshots)
    query_1845 = _query(day, 18, 45)
    receipt_1842 = _query(day, 18, 42)
    query_1915 = _query(day, 19, 15)
    receipt_1902 = _query(day, 19, 2)

    without_receipt = case_output_runtime_table_as_of(
        snapshots, query_1845, horizon=5)
    with_receipt = case_output_runtime_table_as_of(
        snapshots, query_1845, horizon=5,
        verified_receipt_at=receipt_1842)
    late_receipt = case_output_runtime_table_as_of(
        snapshots, query_1915, horizon=5,
        verified_receipt_at=receipt_1902)
    assumed_replay = case_output_table_as_of(
        snapshots, query_1845, horizon=5)

    assert set(without_receipt.corridor) == set(CORRIDORS)
    assert set(with_receipt.corridor) == set(CORRIDORS)
    assert set(late_receipt.corridor) == set(CORRIDORS)
    assert not without_receipt.source_kind.isin(
        RECEIPT_DEPENDENT_SOURCE_KINDS).any()
    assert without_receipt.receipt_verified.eq(False).all()
    assert with_receipt.source_kind.eq("cbr_receipt").all()
    assert with_receipt.receipt_verified.eq(True).all()
    assert pd.to_datetime(with_receipt.receipt_at, utc=True).eq(
        pd.Timestamp(receipt_1842).tz_convert("UTC")).all()
    assert pd.to_datetime(with_receipt.last_source_at, utc=True).ge(
        pd.Timestamp(receipt_1842).tz_convert("UTC")).all()
    assert late_receipt.source_kind.eq("post_receipt_market").all()
    assert pd.to_datetime(late_receipt.last_source_at, utc=True).ge(
        pd.Timestamp(receipt_1902).tz_convert("UTC")).all()
    assert assumed_replay.source_kind.eq("cbr_receipt").all()

    base_local = snapshots.valid_from.dt.tz_convert(MOSCOW)
    base_receipt = snapshots[
        base_local.dt.date.astype(object).eq(day)
        & snapshots.source_kind.eq("cbr_receipt")
    ].sort_values("currency")
    routed = with_receipt.sort_values("corridor")
    for h in HORIZONS:
        if h != 5:
            continue
        np.testing.assert_array_equal(
            routed[f"probability_now_best_h"].to_numpy(),
            base_receipt[f"probability_h{h}"].to_numpy(),
        )
        np.testing.assert_array_equal(
            routed[f"expected_future_cbr_bps_h"].to_numpy(),
            base_receipt[f"expected_future_bps_h{h}"].to_numpy(),
        )
    np.testing.assert_array_equal(
        routed.push_now.to_numpy(), base_receipt.push_now.to_numpy())

    changed = snapshots.copy()
    future = changed.valid_from > pd.Timestamp(query_1845).tz_convert("UTC")
    for column in ("probability_h5", "expected_future_bps_h5"):
        changed.loc[future, column] = 1e9
    prefix_again = case_output_runtime_table_as_of(
        changed, query_1845, horizon=5,
        verified_receipt_at=receipt_1842)
    assert _records(with_receipt) == _records(prefix_again)

    gated_no_event = receipt_gated_snapshots(snapshots, query_1845)
    no_event_local = gated_no_event.valid_from.dt.tz_convert(MOSCOW)
    same_day = no_event_local.dt.date.astype(object).eq(day)
    assert not gated_no_event.loc[same_day].source_kind.isin(
        RECEIPT_DEPENDENT_SOURCE_KINDS).any()

    return {
        "day": day,
        "without_receipt": without_receipt,
        "with_receipt": with_receipt,
        "late_receipt": late_receipt,
        "assumed_replay": assumed_replay,
        "checks": {
            "all_five_corridors_before_receipt": True,
            "no_same_day_receipt_model_without_event": True,
            "verified_event_activates_cbr_row": True,
            "assumed_1830_replaced_by_verified_1842": True,
            "late_receipt_delays_1900_market_update": True,
            "numeric_outputs_and_push_preserved": True,
            "future_row_corruption_prefix_invariant": True,
            "historical_receipts_certified": False,
            "bank_execution_validated": False,
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_audit()
    for name in (
        "without_receipt", "with_receipt", "late_receipt", "assumed_replay",
    ):
        result[name].to_csv(OUT / f"{name}.csv", index=False)
    examples = {
        name: _records(result[name])
        for name in (
            "without_receipt", "with_receipt", "late_receipt", "assumed_replay",
        )
    }
    (OUT / "query_examples.json").write_text(json.dumps(
        examples, ensure_ascii=False, indent=2))
    (OUT / "audit_checks.json").write_text(json.dumps(
        result["checks"], ensure_ascii=False, indent=2))

    sources = [
        BASE / "metadata.json",
        BASE / "snapshots.csv.gz",
        Path("research/temperature_t18_verified_receipt_gate_registered.md"),
        Path("research/temperature_t18_verified_receipt_gate.py"),
        Path("ml/transfer_temperature.py"),
        Path("run_case_output.py"),
    ]
    metadata = {
        "packet": "temperature-T18",
        "example_day": result["day"].isoformat(),
        "predictive_outputs_refit": False,
        "selection_used_open_outcomes": False,
        "production_default_requires_verified_receipt": True,
        "historical_calendar_assumption_explicit_only": True,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({
        "example_day": result["day"].isoformat(),
        "without_receipt_sources": result[
            "without_receipt"].source_kind.value_counts().to_dict(),
        "with_receipt_sources": result[
            "with_receipt"].source_kind.value_counts().to_dict(),
        "late_receipt_sources": result[
            "late_receipt"].source_kind.value_counts().to_dict(),
        **result["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
