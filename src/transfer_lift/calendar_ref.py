"""Holiday calendar for recipient countries.

This is NOT look-ahead: religious and public holiday dates are published years in
advance (Eid al-Adha 2026 was known back in 2019). Look-ahead would mean using
market data from the future; there is none here, only a pre-published calendar.

Ported from the sibling project's ml/calendar_ref.py, which already carries this
exact rationale and holiday set for the same five corridors.
"""
from __future__ import annotations

import datetime as dt

# Eid al-Adha (Kurban Bayram) - date floats on the Gregorian calendar
EID_AL_ADHA = [
    "2018-08-21", "2019-08-11", "2020-07-31", "2021-07-20", "2022-07-09",
    "2023-06-28", "2024-06-16", "2025-06-06", "2026-05-27", "2027-05-16",
]
# Eid al-Fitr (Uraza Bayram)
EID_AL_FITR = [
    "2018-06-15", "2019-06-04", "2020-05-24", "2021-05-13", "2022-05-02",
    "2023-04-21", "2024-04-10", "2025-03-30", "2026-03-20", "2027-03-09",
]


def _fixed(month: int, day: int, years: range) -> list[str]:
    return [f"{y}-{month:02d}-{day:02d}" for y in years]


_YEARS = range(2018, 2028)
HOLIDAYS: dict[str, list[dt.date]] = {
    "eid_adha": [dt.date.fromisoformat(s) for s in EID_AL_ADHA],
    "eid_fitr": [dt.date.fromisoformat(s) for s in EID_AL_FITR],
    "navruz": [dt.date.fromisoformat(s) for s in _fixed(3, 21, _YEARS)],
    "new_year": [dt.date.fromisoformat(s) for s in _fixed(1, 1, _YEARS)],
    "sep_first": [dt.date.fromisoformat(s) for s in _fixed(9, 1, _YEARS)],
}

# Holidays relevant to each corridor (recipient country).
CORRIDOR_HOLIDAYS: dict[str, tuple[str, ...]] = {
    "TJS": ("navruz", "eid_adha", "eid_fitr", "new_year", "sep_first"),
    "UZS": ("navruz", "eid_adha", "eid_fitr", "new_year", "sep_first"),
    "KGS": ("navruz", "eid_adha", "eid_fitr", "new_year", "sep_first"),
    "AMD": ("new_year", "sep_first"),
    "KZT": ("navruz", "eid_adha", "new_year", "sep_first"),
}

PAYDAY_DAYS: tuple[int, ...] = (5, 20)


def days_to_next(day: dt.date, events: list[dt.date]) -> int:
    """Days to the nearest FUTURE calendar event. Calendar is known in advance."""
    future = [e for e in events if e >= day]
    return (future[0] - day).days if future else 999


def days_since_prev(day: dt.date, events: list[dt.date]) -> int:
    past = [e for e in events if e <= day]
    return (day - past[-1]).days if past else 999


def days_to_payday(day: dt.date) -> int:
    """Days to the nearest typical payday (5th or 20th of the month)."""
    for offset in range(0, 32):
        if (day + dt.timedelta(days=offset)).day in PAYDAY_DAYS:
            return offset
    return 99


def days_since_payday(day: dt.date) -> int:
    for offset in range(0, 32):
        if (day - dt.timedelta(days=offset)).day in PAYDAY_DAYS:
            return offset
    return 99
