"""T12: improved unified widget router with an honest 10:00 snapshot."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of
from research.after_publication_ap37_effective import CANDIDATE as PUSH_CANDIDATE
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t8b_unified_router import (
    AP,
    CUTOFFS,
    HORIZONS,
    MOSCOW,
    T3,
    T4,
    T5,
    T6,
    T7B,
    _benefit_columns,
    _probability_columns,
    after_receipt_rows,
    local_timestamp,
)


OUT = Path("results/research/temperature/t12_unified_router")
T10 = Path("results/research/temperature/t10_early1000")
T11 = Path("results/research/temperature/t11_causal_benefit_gate")


def _early_source_timestamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(MOSCOW)
    else:
        stamp = stamp.tz_convert(MOSCOW)
    return stamp.isoformat()


def _stable_pre_receipt_bps(t11, state, h, i):
    key = (f"adaptive__{state}__h{h}"
           if state not in {"premarket", "early_1000"} and h != 20
           else f"prior__{state}__h{h}")
    return float(t11[key][i]), ("causal_adaptive" if key.startswith("adaptive")
                               else "mature_prior")


def pre_receipt_rows(index, t3, t4, t5, t10, t11):
    rows = []
    dates = np.asarray([item[2] for item in index], dtype=object)
    currencies = np.asarray([item[0] for item in index], dtype=object)
    early_available = t10["available"].astype(bool)
    for i, (day, currency) in enumerate(zip(dates, currencies)):
        if day.year < 2024:
            continue
        row = {
            "currency": currency, "valid_from": local_timestamp(day, "0000"),
            "source_at": local_timestamp(day, "0000"),
            "source_kind": "cbr_history", "phase": "premarket",
            "confidence": "limited", "availability_evidence": "calendar_assumed",
            "benefit_source": "mature_prior", "push_now": False,
        }
        _probability_columns(row, lambda h: t4[f"prob__history_hist__h{h}"][i])
        _benefit_columns(row, lambda h: _stable_pre_receipt_bps(
            t11, "premarket", h, i)[0])
        rows.append(row)

        if early_available[i]:
            row = {
                "currency": currency,
                "valid_from": local_timestamp(day, "1000"),
                "source_at": _early_source_timestamp(t10["source_at"][i]),
                "source_kind": "moex_early_prefix", "phase": "market_prefix",
                "confidence": "mature_history",
                "availability_evidence": (
                    "completed_same_day_cny_candle_strict_before_1000"),
                "benefit_source": "mature_prior", "push_now": False,
            }
            _probability_columns(
                row, lambda h: t10[f"routed_probability_h{h}"][i])
            _benefit_columns(row, lambda h: _stable_pre_receipt_bps(
                t11, "early_1000", h, i)[0])
            rows.append(row)

        for cutoff in CUTOFFS:
            state = "cutoff_" + cutoff
            row = {
                "currency": currency,
                "valid_from": local_timestamp(day, cutoff),
                "source_at": local_timestamp(day, cutoff),
                "source_kind": "moex_prefix", "phase": "market_prefix",
                "confidence": "mature_history",
                "availability_evidence": "completed_candles_strict_before_cutoff",
                "benefit_source": "adaptive_h1_h3_h5_h10_prior_h20",
                "push_now": False,
            }
            _probability_columns(
                row, lambda h, state=state: t5[f"prob__{state}__h{h}"][i])
            _benefit_columns(row, lambda h, state=state: _stable_pre_receipt_bps(
                t11, state, h, i)[0])
            rows.append(row)

        for clock in ("1630", "1730"):
            state = "update_" + clock
            row = {
                "currency": currency,
                "valid_from": local_timestamp(day, clock),
                "source_at": local_timestamp(day, clock),
                "source_kind": "post_window_market",
                "phase": "pre_receipt_bridge", "confidence": "mature_history",
                "availability_evidence": "completed_candles_strict_before_cutoff",
                "benefit_source": "adaptive_h1_h3_h5_h10_prior_h20",
                "push_now": False,
            }
            _probability_columns(
                row, lambda h, state=state: t3[f"prob__{state}__h{h}"][i])
            _benefit_columns(row, lambda h, state=state: _stable_pre_receipt_bps(
                t11, state, h, i)[0])
            rows.append(row)
    return rows
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from research.round5_features import load_round5_features
    _matrix, _names, index, _series, *_ = load_round5_features()
    t3, t4, t5, t7b, ap, t10, t11 = (
        loadz(path / "outputs.npz")
        for path in (T3, T4, T5, T7B, AP, T10, T11))
    panel = pd.read_csv(AP / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    rows = pre_receipt_rows(index, t3, t4, t5, t10, t11)
    rows.extend(after_receipt_rows(panel, ap, t7b))
    snapshots = pd.DataFrame(rows)
    snapshots.valid_from = pd.to_datetime(snapshots.valid_from, utc=True)
    snapshots.source_at = pd.to_datetime(snapshots.source_at, utc=True)
    snapshots = snapshots.sort_values(
        ["valid_from", "currency", "source_kind"]).reset_index(drop=True)
    assert not snapshots.duplicated(["currency", "valid_from"]).any()
    assert (snapshots.source_at <= snapshots.valid_from).all()
    snapshots.to_csv(OUT / "snapshots.csv.gz", index=False, compression="gzip")

    local_days = snapshots.valid_from.dt.tz_convert(MOSCOW).dt.date
    last_day = local_days.max()
    full_day = local_days[snapshots.source_kind.eq("cbr_receipt")].max()
    examples = {}
    for label, day, clock in (
            ("premarket", last_day, "0900"),
            ("early_market", last_day, "1015"),
            ("intraday", last_day, "1445"),
            ("after_decision", full_day, "1845"),
            ("after_market_update", full_day, "1930"),
            ("overnight", full_day, "2300")):
        query = dt.datetime(day.year, day.month, day.day,
                            int(clock[:2]), int(clock[2:]), tzinfo=MOSCOW)
        examples[label] = score_snapshot_as_of(snapshots, "KZT", query, 5)
    weekend = last_day
    while weekend.weekday() < 5:
        weekend += dt.timedelta(days=1)
    examples["weekend"] = score_snapshot_as_of(
        snapshots, "KZT", dt.datetime.combine(
            weekend, dt.time(12), tzinfo=MOSCOW), 5)
    (OUT / "query_examples.json").write_text(json.dumps(
        examples, ensure_ascii=False, indent=2))
    sources = [
        *(path / "metadata.json" for path in
          (T3, T4, T5, T7B, AP, T10, T11)),
        Path("research/temperature_t12_unified_router_registered.md"),
        Path("research/temperature_t12_unified_router.py"),
        Path("ml/transfer_temperature.py"),
    ]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T12",
        "snapshot_rows": int(len(snapshots)),
        "early_1000_rows": int(snapshots.source_kind.eq(
            "moex_early_prefix").sum()),
        "first_valid_from": snapshots.valid_from.min().isoformat(),
        "last_valid_from": snapshots.valid_from.max().isoformat(),
        "horizons": list(HORIZONS), "push_candidate": PUSH_CANDIDATE,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    print(json.dumps(examples, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
