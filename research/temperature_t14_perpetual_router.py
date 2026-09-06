"""T14: T12 plus an audited horizon-aware 09:00 perpetual snapshot."""
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


OUT = Path("results/research/temperature/t14_perpetual_router")
T11 = Path("results/research/temperature/t11_causal_benefit_gate")
T12 = Path("results/research/temperature/t12_unified_router")
T13 = Path("results/research/temperature/t13_early_perpetual")
STATE = "perp_0900"
SELECTED = {1: "perp_basis_rank", 3: "perp_basis_rank",
            5: "control", 10: "control", 20: "control"}


def _source_timestamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(MOSCOW)
    else:
        stamp = stamp.tz_convert(MOSCOW)
    return stamp.isoformat()


def early_perpetual_rows(t13, t11):
    dates = np.asarray(t13["dates"], dtype=object)
    currencies = np.asarray(t13["currencies"], dtype=object)
    rows = []
    for i, (day, currency) in enumerate(zip(dates, currencies)):
        if day.year < 2024:
            continue
        usable = {
            h: (bool(t13[f"usable__{STATE}__{SELECTED[h]}__h{h}"][i])
                if SELECTED[h] != "control" else False)
            for h in HORIZONS}
        if not any(usable.values()):
            continue
        market_source = _source_timestamp(t13[f"source_at__{STATE}"][i])
        history_source = local_timestamp(day, "0000")
        row = {
            "currency": currency,
            "valid_from": local_timestamp(day, "0900"),
            "source_at": market_source,
            "source_kind": "mixed_horizon_prefix",
            "phase": "mixed_horizon_prefix",
            "confidence": "mixed_by_horizon",
            "availability_evidence": "horizon_specific",
            "benefit_source": "mature_prior",
            "push_now": False,
        }
        for h in HORIZONS:
            candidate = SELECTED[h]
            if candidate == "control":
                probability = float(t13[f"control__{STATE}__h{h}"][i])
            else:
                probability = float(
                    t13[f"routed__{STATE}__{candidate}__h{h}"][i])
            row[f"probability_h{h}"] = probability
            row[f"expected_future_bps_h{h}"] = float(
                t11[f"prior__premarket__h{h}"][i])
            if usable[h]:
                row[f"source_at_h{h}"] = market_source
                row[f"source_kind_h{h}"] = "moex_perpetual_prefix"
                row[f"phase_h{h}"] = "market_prefix"
                row[f"confidence_h{h}"] = "mature_history"
                row[f"availability_evidence_h{h}"] = (
                    "completed_same_day_cnyrubf_hour_strict_before_0900")
            else:
                row[f"source_at_h{h}"] = history_source
                row[f"source_kind_h{h}"] = "cbr_history"
                row[f"phase_h{h}"] = "premarket"
                row[f"confidence_h{h}"] = "limited"
                row[f"availability_evidence_h{h}"] = "calendar_assumed"
        rows.append(row)
    return rows


def build_snapshots():
    base = pd.read_csv(T12 / "snapshots.csv.gz")
    t13 = loadz(T13 / "outputs.npz")
    t11 = loadz(T11 / "outputs.npz")
    early = pd.DataFrame(early_perpetual_rows(t13, t11))
    snapshots = pd.concat([base, early], ignore_index=True, sort=False)
    snapshots.valid_from = pd.to_datetime(
        snapshots.valid_from, utc=True, format="mixed")
    snapshots.source_at = pd.to_datetime(
        snapshots.source_at, utc=True, format="mixed")
    snapshots = snapshots.sort_values(
        ["valid_from", "currency", "source_kind"]).reset_index(drop=True)
    return snapshots, early


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots, early = build_snapshots()
    assert not snapshots.duplicated(["currency", "valid_from"]).any()
    assert (snapshots.source_at <= snapshots.valid_from).all()
    snapshots.to_csv(OUT / "snapshots.csv.gz", index=False, compression="gzip")

    early_days = pd.to_datetime(early.valid_from, utc=True).dt.tz_convert(MOSCOW)
    example_day = early_days.dt.date.max()
    query = dt.datetime.combine(example_day, dt.time(9, 15), tzinfo=MOSCOW)
    examples = {
        "h1_0915": score_snapshot_as_of(snapshots, "KZT", query, 1),
        "h3_0915": score_snapshot_as_of(snapshots, "KZT", query, 3),
        "h5_0915": score_snapshot_as_of(snapshots, "KZT", query, 5),
    }
    weekend = example_day
    while weekend.weekday() < 5:
        weekend += dt.timedelta(days=1)
    examples["weekend_h1"] = score_snapshot_as_of(
        snapshots, "KZT",
        dt.datetime.combine(weekend, dt.time(12), tzinfo=MOSCOW), 1)
    (OUT / "query_examples.json").write_text(json.dumps(
        examples, ensure_ascii=False, indent=2))

    sources = [
        T11 / "metadata.json", T11 / "outputs.npz",
        T12 / "metadata.json", T12 / "snapshots.csv.gz",
        T13 / "metadata.json", T13 / "outputs.npz", T13 / "selection.csv",
        Path("research/temperature_t14_perpetual_router_registered.md"),
        Path("research/temperature_t14_perpetual_router.py"),
        Path("ml/transfer_temperature.py"),
    ]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T14",
        "snapshot_rows": int(len(snapshots)),
        "new_0900_rows": int(len(early)),
        "selected_0900_probability_heads": SELECTED,
        "magnitude_0900": "T11 mature premarket prior",
        "horizon_specific_provenance": True,
        "historical_exchange_availability_certified": False,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    print(json.dumps(examples, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
