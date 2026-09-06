"""T16: T14 plus audited benefit-only perpetual updates at 20:00-23:00."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t8b_unified_router import HORIZONS, MOSCOW, local_timestamp


OUT = Path("results/research/temperature/t16_evening_router")
T14 = Path("results/research/temperature/t14_perpetual_router")
T15 = Path("results/research/temperature/t15_evening_perpetual")
AP = Path("results/research/after_publication/ap51_benefit")
CLOCKS = ("2000", "2100", "2200", "2300")


def _local_source(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(MOSCOW)
    else:
        stamp = stamp.tz_convert(MOSCOW)
    return stamp.isoformat()


def _selected(selection, clock, h):
    part = selection[
        selection.kind.eq("benefit")
        & selection.clock.eq("perp_" + clock)
        & selection.h.eq(h)]
    if len(part) != 1:
        raise AssertionError((clock, h, len(part)))
    return bool(part.iloc[0].adopted)


def _probability_provenance(row, receipt_at, h, carried_at=None):
    if h == 1:
        return {
            "source_at": receipt_at,
            "source_kind": "cbr_receipt",
            "phase": "after_new_cbr",
            "confidence": "mature_history",
            "availability": "calendar_assumed",
        }
    return {
        "source_at": carried_at if carried_at is not None else row["valid_from"],
        "source_kind": "post_receipt_market",
        "phase": "after_new_cbr_market_update",
        "confidence": "mature_history",
        "availability": "calendar_assumed_receipt_plus_completed_candles",
    }


def _set_probability_provenance(row, receipt_at, h, carried_at=None):
    provenance = _probability_provenance(row, receipt_at, h, carried_at)
    row[f"source_at_h{h}"] = provenance["source_at"]
    row[f"source_kind_h{h}"] = provenance["source_kind"]
    row[f"phase_h{h}"] = provenance["phase"]
    row[f"confidence_h{h}"] = provenance["confidence"]
    row[f"availability_evidence_h{h}"] = provenance["availability"]


def _set_benefit_provenance(row, h, source_at, kind, availability):
    row[f"benefit_source_at_h{h}"] = source_at
    row[f"benefit_source_kind_h{h}"] = kind
    row[f"benefit_availability_evidence_h{h}"] = availability


def build_snapshots():
    snapshots = pd.read_csv(T14 / "snapshots.csv.gz")
    snapshots.valid_from = pd.to_datetime(snapshots.valid_from, utc=True)
    snapshots.source_at = pd.to_datetime(snapshots.source_at, utc=True)
    for h in HORIZONS:
        for field in ("benefit_source_at", "benefit_source_kind",
                      "benefit_availability_evidence"):
            column = f"{field}_h{h}"
            if column not in snapshots:
                snapshots[column] = pd.Series(
                    [None] * len(snapshots), dtype=object)
    t15 = loadz(T15 / "outputs.npz")
    selection = pd.read_csv(T15 / "selection.csv")
    panel = pd.read_csv(AP / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    additions = []
    base_lookup = {
        (currency, valid_from): i
        for i, (currency, valid_from) in enumerate(zip(
            snapshots.currency, snapshots.valid_from))
    }

    for i, item in panel.iterrows():
        day, currency = item.date, item.currency
        if day.year < 2024:
            continue
        receipt_at = pd.Timestamp(item.decision_at).tz_convert(MOSCOW).isoformat()
        valid_2000 = pd.Timestamp(local_timestamp(day, "2000")).tz_convert("UTC")
        base_i = base_lookup.get((currency, valid_2000))
        if base_i is None:
            raise AssertionError((day, currency, "missing T14 20:00"))
        base_row = snapshots.iloc[base_i].copy()
        for h in HORIZONS:
            _set_probability_provenance(base_row, receipt_at, h)
            adopted = h in (3, 5, 10, 20) and _selected(
                selection, "2000", h)
            usable = (bool(t15[f"benefit_usable__perp_2000__h{h}"][i])
                      if h in (3, 5, 10, 20) else False)
            if adopted and usable:
                base_row[f"expected_future_bps_h{h}"] = float(
                    t15[f"routed_benefit__perp_2000__h{h}"][i])
                source = _local_source(t15["dual_source_at__perp_2000"][i])
                _set_benefit_provenance(
                    base_row, h, source, "post_receipt_perpetual",
                    "completed_dual_perpetual_prefix_strict_before_2000")
            elif h == 1:
                _set_benefit_provenance(
                    base_row, h, receipt_at, "cbr_receipt",
                    "calendar_assumed")
            else:
                _set_benefit_provenance(
                    base_row, h, base_row["valid_from"],
                    "post_receipt_market",
                    "calendar_assumed_receipt_plus_completed_candles")
        snapshots.loc[base_i] = base_row

        for clock in CLOCKS[1:]:
            state = "perp_" + clock
            if not bool(t15[f"dual_available__{state}"][i]):
                continue
            source = _local_source(t15[f"dual_source_at__{state}"][i])
            row = {
                "currency": currency,
                "valid_from": local_timestamp(day, clock),
                "source_at": source,
                "source_kind": "post_receipt_perpetual",
                "phase": "after_new_cbr_perpetual_benefit_update",
                "confidence": "mixed_by_output",
                "availability_evidence": (
                    "completed_dual_perpetual_prefix_strict_before_" + clock),
                "benefit_source": "horizon_specific",
                "push_now": False,
            }
            for h in HORIZONS:
                if h == 1:
                    row[f"probability_h{h}"] = float(
                        t15["known_probability_h1"][i])
                    row[f"expected_future_bps_h{h}"] = float(
                        t15["known_future_bps_h1"][i])
                else:
                    row[f"probability_h{h}"] = float(
                        t15[f"control_probability__{state}__h{h}"][i])
                    adopted = _selected(selection, clock, h)
                    usable = bool(t15[f"benefit_usable__{state}__h{h}"][i])
                    key = (f"routed_benefit__{state}__h{h}"
                           if adopted and usable
                           else f"control_benefit__{state}__h{h}")
                    row[f"expected_future_bps_h{h}"] = float(t15[key][i])
                _set_probability_provenance(
                    row, receipt_at, h, local_timestamp(day, "2000"))
                if h == 1:
                    _set_benefit_provenance(
                        row, h, receipt_at, "cbr_receipt",
                        "calendar_assumed")
                elif _selected(selection, clock, h) and bool(
                        t15[f"benefit_usable__{state}__h{h}"][i]):
                    _set_benefit_provenance(
                        row, h, source, "post_receipt_perpetual",
                        "completed_dual_perpetual_prefix_strict_before_" + clock)
                else:
                    _set_benefit_provenance(
                        row, h, local_timestamp(day, "2000"),
                        "post_receipt_market",
                        "calendar_assumed_receipt_plus_completed_candles")
            additions.append(row)

    if additions:
        snapshots = pd.concat(
            [snapshots, pd.DataFrame(additions)], ignore_index=True, sort=False)
    snapshots.valid_from = pd.to_datetime(
        snapshots.valid_from, utc=True, format="mixed")
    snapshots.source_at = pd.to_datetime(
        snapshots.source_at, utc=True, format="mixed")
    snapshots = snapshots.sort_values(
        ["valid_from", "currency", "source_kind"]).reset_index(drop=True)
    return snapshots, pd.DataFrame(additions)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots, additions = build_snapshots()
    assert not snapshots.duplicated(["currency", "valid_from"]).any()
    assert (snapshots.source_at <= snapshots.valid_from).all()
    snapshots.to_csv(OUT / "snapshots.csv.gz", index=False, compression="gzip")

    local = snapshots.valid_from.dt.tz_convert(MOSCOW)
    full_day = local[snapshots.source_kind.eq("cbr_receipt")].dt.date.max()
    examples = {}
    for label, clock, horizon in (
        ("at_2015_h3", "2015", 3),
        ("at_2115_h5", "2115", 5),
        ("at_2315_h3", "2315", 3),
        ("at_2315_h10_control", "2315", 10),
    ):
        query = dt.datetime(
            full_day.year, full_day.month, full_day.day,
            int(clock[:2]), int(clock[2:]), tzinfo=MOSCOW)
        examples[label] = score_snapshot_as_of(
            snapshots, "KZT", query, horizon)
    (OUT / "query_examples.json").write_text(json.dumps(
        examples, ensure_ascii=False, indent=2))
    sources = [
        T14 / "metadata.json", T14 / "snapshots.csv.gz",
        T15 / "metadata.json", T15 / "outputs.npz", T15 / "selection.csv",
        Path("research/temperature_t16_evening_router_registered.md"),
        Path("research/temperature_t16_evening_router.py"),
        Path("ml/transfer_temperature.py"),
    ]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T16",
        "snapshot_rows": int(len(snapshots)),
        "new_evening_rows": int(len(additions)),
        "probability_replacements": 0,
        "benefit_replacements": {
            "2000": [3], "2100": [3, 5],
            "2200": [3, 5], "2300": [3, 5],
        },
        "separate_probability_benefit_provenance": True,
        "historical_exchange_availability_certified": False,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }, indent=2))
    print(json.dumps(examples, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
