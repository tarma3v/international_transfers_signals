import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from ml.data import Series
from research.after_publication_clock import MOSCOW, RateRecord, calendar_assumed_records, snapshot


def instant(day, hour=18):
    return dt.datetime.fromisoformat(day).replace(hour=hour, tzinfo=MOSCOW)


def records():
    return (
        RateRecord("TJS", dt.date(2024, 2, 2), instant("2024-02-01"), 8.2, "observed_receipt", 0),
        RateRecord("TJS", dt.date(2024, 2, 3), instant("2024-02-02"), 8.27478, "observed_receipt", 1),
        RateRecord("TJS", dt.date(2024, 2, 6), instant("2024-02-05"), 8.32596, "observed_receipt", 2),
    )


def test_saturday_cannot_see_monday_publication():
    state = snapshot(records(), instant("2024-02-03"))
    assert state.current_effective.value_rub_per_unit == 8.27478
    assert state.latest_announced.source_index == 1
    assert state.next_announced is None
    assert not state.calendar_assumption_used


def test_monday_after_receipt_has_current_and_tomorrow_separate():
    state = snapshot(records(), instant("2024-02-05"))
    assert state.current_effective.source_index == 1
    assert state.next_announced.source_index == 2
    assert state.latest_announced.source_index == 2


def test_just_before_receipt_excludes_next_and_utc_agrees():
    decision = instant("2024-02-05") - dt.timedelta(microseconds=1)
    assert snapshot(records(), decision).next_announced is None
    state = snapshot(records(), decision.astimezone(dt.timezone.utc))
    assert state.current_effective.source_index == 1
    assert state.next_announced is None


def test_future_corruption_and_permutation_leave_state_unchanged():
    rows = records()
    corrupt = (replace(rows[-1], value_rub_per_unit=999999), rows[0], rows[1])
    assert snapshot(rows, instant("2024-02-03")) == snapshot(corrupt, instant("2024-02-03"))


def test_receipt_not_effective_date_controls_knowledge():
    rows = list(records())
    rows[-1] = replace(rows[-1], received_at=instant("2024-02-07"))
    state = snapshot(rows, instant("2024-02-06"))
    assert state.current_effective.source_index == 1
    assert state.latest_announced.source_index == 1


def test_out_of_order_old_record_does_not_replace_new_rate():
    rows = list(records())
    rows[0] = replace(rows[0], received_at=instant("2024-02-06"))
    assert snapshot(rows, instant("2024-02-06")).current_effective.source_index == 2


def test_revised_same_effective_date_uses_latest_receipt():
    rows = records()
    revision = replace(rows[1], received_at=instant("2024-02-03", 12), value_rub_per_unit=8.3)
    state = snapshot(rows + (revision,), instant("2024-02-03"))
    assert state.current_effective.value_rub_per_unit == 8.3


def test_calendar_assumption_cannot_be_mistaken_for_observed():
    series = Series("TJS", np.array([dt.date(2024, 2, 3), dt.date(2024, 2, 6)]), np.array([8.27478,8.32596]))
    rows = calendar_assumed_records(series)
    assert rows[0].received_at == instant("2024-02-02")
    assert rows[1].received_at == instant("2024-02-05")
    state = snapshot(rows, instant("2024-02-03"))
    assert state.calendar_assumption_used and state.next_announced is None
    assert all(r.evidence == "calendar_assumed" for r in rows)


def test_empty_known_prefix_and_mixed_or_naive_inputs():
    state = snapshot(records(), instant("2024-01-01"))
    assert state.current_effective is None and state.latest_announced is None
    with pytest.raises(ValueError):
        snapshot(records(), dt.datetime(2024,2,3,18))
    with pytest.raises(ValueError):
        snapshot((), instant("2024-02-03"))
    with pytest.raises(ValueError):
        snapshot(records() + (replace(records()[0],currency="KZT"),), instant("2024-02-03"))
    with pytest.raises(ValueError):
        replace(records()[0],received_at=dt.datetime(2024,2,3))
    with pytest.raises(ValueError):
        replace(records()[0],evidence="inferred_but_trust_me")
