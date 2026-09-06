"""T17: remove synthetic spot checkpoints and use observed candle timestamps."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

from ml.targets import HORIZONS, build_targets
from ml.transfer_temperature import score_snapshot_as_of
from research.round5_features import load_round5_features
from research.round6_moex_spot_1530_features import (
    SESSION_START,
    _arrays,
    load_spot_1530_history,
)
from research.temperature_t8b_unified_router import MOSCOW


OUT = Path("results/research/temperature/t17_spot_availability_repair")
BASE = Path("results/research/temperature/t16_evening_router")
DATA = Path("data/moex_spot_fx_10min_2022_2026.json")
SPOT_SOURCE_KINDS = {
    "moex_prefix", "post_window_market", "post_receipt_market",
}
SPOT_PROVENANCE_KINDS = SPOT_SOURCE_KINDS | {"moex_early_prefix"}
OBSERVED_EVIDENCE = (
    "observed_completed_same_day_cny_spot_strict_before_valid_from"
)


def load_base():
    frame = pd.read_csv(BASE / "snapshots.csv.gz")
    frame.valid_from = pd.to_datetime(frame.valid_from, utc=True)
    frame.source_at = pd.to_datetime(frame.source_at, utc=True)
    return frame


def _observed_source(item, day, clock):
    start_time = dt.datetime.combine(day, SESSION_START)
    cutoff = dt.datetime.combine(day, clock)
    start = int(np.searchsorted(item["begin"], start_time, side="left"))
    stop = int(np.searchsorted(item["end"], cutoff, side="left"))
    rows = np.arange(start, stop, dtype=int)
    if len(rows):
        rows = rows[
            (item["begin"][rows] >= start_time)
            & (item["end"][rows] < cutoff)
        ]
    if not len(rows):
        return pd.NaT
    value = pd.Timestamp(item["end"][rows[-1]])
    return value.tz_localize(MOSCOW).tz_convert("UTC")


def observed_sources(frame, history):
    item = _arrays(history)["CNYRUB_TOM"]
    local = frame.valid_from.dt.tz_convert(MOSCOW)
    candidate = frame.source_kind.isin(SPOT_SOURCE_KINDS)
    keys = sorted({
        (stamp.date(), stamp.time().replace(tzinfo=None))
        for stamp in local[candidate]
    })
    lookup = {
        key: _observed_source(item, key[0], key[1])
        for key in keys
    }
    source = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for i in frame.index[candidate]:
        stamp = local.iloc[i]
        key = (stamp.date(), stamp.time().replace(tzinfo=None))
        source.loc[i] = lookup[key]
    details = pd.DataFrame({
        "base_index": frame.index[candidate].astype(int),
        "currency": frame.loc[candidate, "currency"].to_numpy(),
        "valid_from": frame.loc[candidate, "valid_from"].to_numpy(),
        "source_kind": frame.loc[candidate, "source_kind"].to_numpy(),
        "observed_source_at": source.loc[candidate].to_numpy(),
    })
    local_candidate = local[candidate]
    details["local_date"] = local_candidate.dt.date.to_numpy()
    details["clock"] = local_candidate.dt.strftime("%H:%M").to_numpy()
    details["year"] = local_candidate.dt.year.to_numpy()
    details["weekday"] = local_candidate.dt.day_name().to_numpy()
    details["observed"] = details.observed_source_at.notna()
    return source, details


def repair_snapshots(base, history):
    frame = base.copy()
    observed, details = observed_sources(frame, history)
    candidate = frame.source_kind.isin(SPOT_SOURCE_KINDS)
    keep = ~candidate | observed.notna()
    repaired = frame.loc[keep].copy()
    kept_spot = repaired.index[repaired.source_kind.isin(SPOT_SOURCE_KINDS)]
    repaired.loc[kept_spot, "source_at"] = observed.loc[kept_spot]
    repaired.loc[kept_spot, "availability_evidence"] = OBSERVED_EVIDENCE

    for h in HORIZONS:
        kind = f"source_kind_h{h}"
        source = f"source_at_h{h}"
        evidence = f"availability_evidence_h{h}"
        if kind in repaired and source in repaired:
            routed = repaired[kind].isin(SPOT_PROVENANCE_KINDS)
            routed &= repaired.index.isin(kept_spot)
            repaired.loc[routed, source] = observed.loc[repaired.index[routed]]
            if evidence in repaired:
                repaired.loc[routed, evidence] = OBSERVED_EVIDENCE

        benefit_kind = f"benefit_source_kind_h{h}"
        benefit_source = f"benefit_source_at_h{h}"
        benefit_evidence = f"benefit_availability_evidence_h{h}"
        if benefit_kind in repaired and benefit_source in repaired:
            routed = repaired[benefit_kind].isin(SPOT_PROVENANCE_KINDS)
            routed &= repaired.index.isin(kept_spot)
            repaired.loc[routed, benefit_source] = observed.loc[
                repaired.index[routed]]
            if benefit_evidence in repaired:
                repaired.loc[routed, benefit_evidence] = OBSERVED_EVIDENCE

    repaired = repaired.sort_values(
        ["valid_from", "currency", "source_kind"]
    ).reset_index(drop=True)
    return repaired, details


def _ece(y, p):
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, 9)
    return float(sum(
        np.mean(bins == i) * abs(float(p[bins == i].mean())
                                 - float(y[bins == i].mean()))
        for i in range(10) if np.any(bins == i)
    ))


def _metric(y, p):
    valid = np.isfinite(y) & np.isfinite(p)
    y = np.asarray(y[valid], dtype=int)
    p = np.clip(np.asarray(p[valid], dtype=float), 1e-6, 1.0 - 1e-6)
    row = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
        "ece": _ece(y, p) if len(y) else np.nan,
    }
    if len(y) and np.unique(y).size == 2:
        row["auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
    else:
        row["auc"] = row["average_precision"] = np.nan
    return row


def _target_lookup():
    _x, _names, index, series, *_ = load_round5_features()
    targets = build_targets(series, index)
    lookup = {}
    for i, (currency, _position, day) in enumerate(index):
        lookup[(currency, day)] = {
            h: float(targets[f"fav_h{h}"][i]) for h in HORIZONS
        }
    return lookup


def _fallback_probabilities(repaired, dropped, h):
    result = np.full(len(dropped), np.nan)
    target_times = pd.to_datetime(dropped.valid_from, utc=True).astype("int64")
    for currency in sorted(dropped.currency.unique()):
        source = repaired[repaired.currency.eq(currency)].sort_values(
            "valid_from")
        times = source.valid_from.astype("int64").to_numpy()
        p = pd.to_numeric(source[f"probability_h{h}"], errors="coerce").to_numpy()
        mask = dropped.currency.eq(currency).to_numpy()
        positions = np.searchsorted(times, target_times[mask].to_numpy(), side="right") - 1
        valid = positions >= 0
        values = np.full(mask.sum(), np.nan)
        values[valid] = p[positions[valid]]
        result[mask] = values
    return result


def diagnostic_metrics(base, repaired, details):
    dropped = details.loc[~details.observed].copy().reset_index(drop=True)
    lookup = _target_lookup()
    rows, reliability = [], []
    for h in HORIZONS:
        old = pd.to_numeric(
            base.loc[dropped.base_index, f"probability_h{h}"], errors="coerce"
        ).to_numpy()
        held = _fallback_probabilities(repaired, dropped, h)
        y = np.asarray([
            lookup.get((currency, day), {}).get(h, np.nan)
            for currency, day in zip(dropped.currency, dropped.local_date)
        ], dtype=float)
        for period, years in (
            ("screen_2024", {2024}),
            ("opened_2025", {2025}),
            ("opened_2026", {2026}),
            ("open_all", {2024, 2025, 2026}),
        ):
            scope = dropped.year.isin(years).to_numpy()
            for model, probability in (("synthetic", old), ("held", held)):
                rows.append({
                    "period": period,
                    "h": h,
                    "model": model,
                    **_metric(y[scope], probability[scope]),
                    "mean_abs_change_temperature": float(np.nanmean(
                        np.abs(old[scope] - held[scope]) * 100.0
                    )),
                })
            valid = scope & np.isfinite(y) & np.isfinite(held)
            if valid.any():
                edges = np.linspace(0.0, 1.0, 11)
                ids = np.minimum(np.searchsorted(
                    edges, held[valid], side="right") - 1, 9)
                for bin_id in range(10):
                    chosen = ids == bin_id
                    if chosen.any():
                        reliability.append({
                            "period": period, "h": h, "model": "held",
                            "bin_left": edges[bin_id],
                            "bin_right": edges[bin_id + 1],
                            "n": int(chosen.sum()),
                            "predicted": float(held[valid][chosen].mean()),
                            "actual": float(y[valid][chosen].mean()),
                        })
    return pd.DataFrame(rows), pd.DataFrame(reliability)


def _query_examples(repaired, details):
    dropped = details.loc[~details.observed].copy()
    weekend = dropped[
        pd.to_datetime(dropped.local_date).dt.dayofweek.ge(5)
    ]
    example = weekend.iloc[0] if len(weekend) else dropped.iloc[0]
    query = pd.Timestamp(example.valid_from).tz_convert(MOSCOW).to_pydatetime()
    query += dt.timedelta(minutes=1)
    return {
        "missing_spot_hold_h5": score_snapshot_as_of(
            repaired, str(example.currency), query, 5),
        "missing_spot_planned_clock": str(example.clock),
        "missing_spot_local_date": str(example.local_date),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base()
    history, digest = load_spot_1530_history()
    repaired, details = repair_snapshots(base, history)
    if repaired.duplicated(["currency", "valid_from"]).any():
        raise AssertionError("duplicate repaired snapshot")
    if not (repaired.source_at <= repaired.valid_from).all():
        raise AssertionError("future row-level source")
    repaired.to_csv(OUT / "snapshots.csv.gz", index=False, compression="gzip")

    availability = details.groupby(
        ["clock", "source_kind", "year", "weekday", "observed"],
        dropna=False,
    ).size().rename("rows").reset_index()
    availability.to_csv(OUT / "availability_by_clock_year_weekday.csv", index=False)
    metrics, reliability = diagnostic_metrics(base, repaired, details)
    metrics.to_csv(OUT / "missing_day_probability_metrics.csv", index=False)
    reliability.to_csv(OUT / "missing_day_reliability_bins.csv", index=False)
    examples = _query_examples(repaired, details)
    (OUT / "query_examples.json").write_text(json.dumps(
        examples, ensure_ascii=False, indent=2))

    sources = [
        BASE / "metadata.json", BASE / "snapshots.csv.gz", DATA,
        Path("research/temperature_t17_spot_availability_repair_registered.md"),
        Path("research/temperature_t17_spot_availability_repair.py"),
        Path("ml/transfer_temperature.py"),
    ]
    dropped = details.loc[~details.observed]
    (OUT / "metadata.json").write_text(json.dumps({
        "packet": "temperature-T17",
        "base_rows": int(len(base)),
        "snapshot_rows": int(len(repaired)),
        "planned_spot_rows": int(len(details)),
        "observed_spot_rows": int(details.observed.sum()),
        "dropped_synthetic_spot_rows": int((~details.observed).sum()),
        "dropped_unique_currency_dates": int(dropped[
            ["currency", "local_date"]].drop_duplicates().shape[0]),
        "actual_spot_source_timestamps": True,
        "probabilities_expected_bps_push_unchanged_on_kept_rows": True,
        "repair_selected_by_metrics": False,
        "open_diagnostics_only": True,
        "payload_sha256": digest,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }, indent=2))
    print(json.dumps({
        "rows": len(repaired),
        "planned_spot": len(details),
        "observed_spot": int(details.observed.sum()),
        "dropped": int((~details.observed).sum()),
        "query_example": examples,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
