"""Explicit as-of contract; calendar-assumed receipts are never verified receipts.

This module does not train or score a model. Inference from an effective date
is a labelled research assumption, never evidence of the actual release clock.
All snapshots are for a single currency and use Moscow dates for effectiveness.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Literal
from zoneinfo import ZoneInfo

MOSCOW = ZoneInfo("Europe/Moscow")
Evidence = Literal["observed_receipt", "calendar_assumed"]


@dataclass(frozen=True)
class RateRecord:
    currency: str
    effective_date: dt.date
    received_at: dt.datetime
    value_rub_per_unit: float
    evidence: Evidence
    source_index: int

    def __post_init__(self):
        import math
        if self.received_at.tzinfo is None or self.received_at.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware")
        if self.evidence not in ("observed_receipt", "calendar_assumed"):
            raise ValueError("Unknown availability evidence")
        if not self.currency or self.source_index < 0:
            raise ValueError("Invalid source identity")
        if not math.isfinite(self.value_rub_per_unit) or self.value_rub_per_unit <= 0:
            raise ValueError("Rate must be positive finite RUB per unit")


@dataclass(frozen=True)
class RateSnapshot:
    currency: str
    decision_at: dt.datetime
    current_effective: RateRecord | None
    latest_announced: RateRecord | None
    next_announced: RateRecord | None
    calendar_assumption_used: bool


def snapshot(records: Iterable[RateRecord], decision_at: dt.datetime) -> RateSnapshot:
    """Only received records may affect the numeric state, including weekends.

    An empty known prefix is valid. An empty total input is rejected because
    its currency is unspecified. Later-arriving old records cannot replace a
    more recent effective date. Revisions of one effective date use last receipt.
    """
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise ValueError("decision_at must be timezone-aware")
    records = tuple(records)
    currencies = {r.currency for r in records}
    if len(currencies) != 1:
        raise ValueError("snapshot requires exactly one currency")
    currency = next(iter(currencies))
    known = [r for r in records if r.received_at <= decision_at]
    key = lambda r: (r.effective_date, r.received_at, r.source_index)
    day = decision_at.astimezone(MOSCOW).date()
    current = max((r for r in known if r.effective_date <= day), key=key, default=None)
    latest = max(known, key=key, default=None)
    future_dates = [r.effective_date for r in known if r.effective_date > day]
    first_future = min(future_dates) if future_dates else None
    upcoming = max((r for r in known if r.effective_date == first_future), key=key, default=None)
    return RateSnapshot(currency, decision_at, current, latest, upcoming,
                        any(r.evidence == "calendar_assumed" for r in known))


def calendar_assumed_records(series, assumed_time: dt.time = dt.time(18)) -> tuple[RateRecord, ...]:
    """CBR ordinary date rule plus an explicit assumed Moscow clock time.

    All rows are tagged calendar_assumed. No caller can label the inferred
    receipt observed via this function. No financial values are shifted.
    """
    if assumed_time.tzinfo is not None:
        raise ValueError("Pass a naive Moscow time, not a timezone-bearing time")
    return tuple(
        RateRecord(series.code, day,
                   dt.datetime.combine(day-dt.timedelta(days=1), assumed_time, MOSCOW),
                   float(value), "calendar_assumed", i)
        for i, (day, value) in enumerate(zip(series.dates, series.values))
    )
