import datetime as dt

from research.temperature_t8b_unified_router import local_timestamp


def test_local_timestamp_is_explicitly_moscow_aware():
    stamp = local_timestamp(dt.date(2026, 9, 4), '1520')
    assert stamp == '2026-09-04T15:20:00+03:00'
