import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from ml.transfer_temperature import (
    case_output_as_of,
    case_output_table_as_of,
    score_snapshot_as_of,
)


def test_router_respects_validity_and_never_uses_future_snapshot():
    moscow = ZoneInfo('Europe/Moscow')
    snapshots = pd.DataFrame({
        'currency': ['KZT', 'KZT', 'KZT'],
        'valid_from': ['2026-09-04T09:00:00+03:00',
                       '2026-09-04T15:30:00+03:00',
                       '2026-09-04T18:30:00+03:00'],
        'valid_until': ['2026-09-04T15:30:00+03:00',
                        '2026-09-04T18:30:00+03:00',
                        '2026-09-05T00:00:00+03:00'],
        'source_at': ['2026-09-03T18:30:00+03:00',
                      '2026-09-04T15:20:00+03:00',
                      '2026-09-04T18:30:00+03:00'],
        'source_kind': ['cbr_history', 'moex_prefix', 'cbr_receipt'],
        'phase': ['premarket', 'market_prefix', 'after_new_cbr'],
        'confidence': ['limited', 'mature_history', 'mature_history'],
        'probability_h5': [.4, .7, .9],
        'push_now': [False, True, False],
    })
    result = score_snapshot_as_of(
        snapshots, 'KZT', dt.datetime(2026, 9, 4, 17, tzinfo=moscow), 5)
    assert result['probability_now_best_h'] == .7
    assert result['phase'] == 'market_prefix'
    assert result['last_source_at'].startswith('2026-09-04T15:20:00')


def test_router_does_not_reuse_expired_after_receipt_reference_next_day():
    moscow = ZoneInfo('Europe/Moscow')
    snapshots = pd.DataFrame({
        'currency': ['KZT'],
        'valid_from': ['2026-09-04T18:30:00+03:00'],
        'valid_until': ['2026-09-05T00:00:00+03:00'],
        'source_at': ['2026-09-04T18:30:00+03:00'],
        'source_kind': ['cbr_receipt'],
        'phase': ['after_new_cbr'],
        'probability_h1': [1.],
    })
    result = score_snapshot_as_of(
        snapshots, 'KZT', dt.datetime(2026, 9, 5, 9, tzinfo=moscow), 1)
    assert result is None


def test_case_output_has_required_schema_and_past_present_copy():
    moscow = ZoneInfo('Europe/Moscow')
    snapshots = pd.DataFrame({
        'currency': ['KZT'],
        'valid_from': ['2026-09-04T15:30:00+03:00'],
        'valid_until': ['2026-09-04T18:30:00+03:00'],
        'source_at': ['2026-09-04T15:19:59+03:00'],
        'source_kind': ['moex_prefix'],
        'phase': ['market_prefix'],
        'confidence': ['mature_history'],
        'availability_evidence': ['observed_completed_candle'],
        'probability_h5': [.73],
        'expected_future_bps_h5': [80.],
        'push_now': [True],
    })
    result = case_output_as_of(
        snapshots, 'KZT', dt.datetime(2026, 9, 4, 16, tzinfo=moscow), 5)
    required = {
        'date', 'corridor', 'indicator', 'direction', 'strength',
        'indicator_speed', 'recommended_scenario',
    }
    assert required <= result.keys()
    assert result['direction'] == 'supports_current_moment'
    assert result['indicator_speed'] == 'fast_intraday'
    assert result['recommended_scenario'] == 'sparse_push_and_fresh_widget'
    forbidden = ('подожд', 'будет', 'гарант', 'переводи')
    assert not any(word in result['label'].lower() for word in forbidden)


def test_case_output_table_returns_each_available_corridor():
    moscow = ZoneInfo('Europe/Moscow')
    currencies = ['AMD', 'KGS', 'KZT', 'TJS', 'UZS']
    snapshots = pd.DataFrame({
        'currency': currencies,
        'valid_from': ['2026-09-04T18:30:00+03:00'] * len(currencies),
        'source_at': ['2026-09-04T18:30:00+03:00'] * len(currencies),
        'source_kind': ['cbr_receipt'] * len(currencies),
        'phase': ['after_new_cbr'] * len(currencies),
        'confidence': ['mature_history'] * len(currencies),
        'availability_evidence': ['observed_receipt'] * len(currencies),
        'probability_h5': [.5] * len(currencies),
        'expected_future_bps_h5': [0.] * len(currencies),
        'push_now': [False] * len(currencies),
    })
    result = case_output_table_as_of(
        snapshots, dt.datetime(2026, 9, 4, 19, tzinfo=moscow), 5)
    assert set(result.corridor) == set(currencies)
    assert result.indicator_speed.eq('slow_daily_publication').all()
