import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ml.transfer_temperature import (
    case_output_as_of,
    case_output_runtime_table_as_of,
    case_output_table_as_of,
    receipt_gated_snapshots,
    score_runtime_as_of,
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


def _receipt_gate_snapshots():
    currencies = ['AMD', 'KGS', 'KZT', 'TJS', 'UZS']
    rows = []
    for currency in currencies:
        common = {
            'currency': currency,
            'confidence': 'mature_history',
            'probability_h5': .6,
            'expected_future_bps_h5': 10.,
            'push_now': False,
        }
        rows.extend([
            {
                **common,
                'valid_from': '2026-09-04T17:30:00+03:00',
                'source_at': '2026-09-04T17:19:59+03:00',
                'source_kind': 'post_window_market',
                'phase': 'pre_receipt_bridge',
                'availability_evidence': 'observed_completed_candle',
            },
            {
                **common,
                'valid_from': '2026-09-04T18:30:00+03:00',
                'source_at': '2026-09-04T18:30:00+03:00',
                'source_kind': 'cbr_receipt',
                'phase': 'after_new_cbr',
                'availability_evidence': 'calendar_assumed',
                'probability_h5': .8,
                'expected_future_bps_h5': 80.,
                'push_now': True,
            },
            {
                **common,
                'valid_from': '2026-09-04T19:00:00+03:00',
                'source_at': '2026-09-04T18:59:59+03:00',
                'source_kind': 'post_receipt_market',
                'phase': 'after_new_cbr_market_update',
                'availability_evidence': 'observed_completed_candle',
                'probability_h5': .82,
                'expected_future_bps_h5': 85.,
            },
        ])
    return pd.DataFrame(rows)


def test_runtime_gate_does_not_invent_calendar_receipt():
    moscow = ZoneInfo('Europe/Moscow')
    query = dt.datetime(2026, 9, 4, 18, 45, tzinfo=moscow)
    table = case_output_runtime_table_as_of(
        _receipt_gate_snapshots(), query, horizon=5)
    assert len(table) == 5
    assert table.source_kind.eq('post_window_market').all()
    assert ~table.receipt_verified.any()
    assert table.last_source_at.str.startswith('2026-09-04T17:19:59').all()


def test_runtime_gate_activates_only_at_verified_event_time():
    moscow = ZoneInfo('Europe/Moscow')
    query = dt.datetime(2026, 9, 4, 18, 45, tzinfo=moscow)
    receipt = dt.datetime(2026, 9, 4, 18, 42, tzinfo=moscow)
    result = score_runtime_as_of(
        _receipt_gate_snapshots(), 'KZT', query, 5,
        verified_receipt_at=receipt)
    assert result['source_kind'] == 'cbr_receipt'
    assert result['probability_now_best_h'] == .8
    assert result['receipt_verified'] is True
    assert result['receipt_at'].startswith('2026-09-04T18:42:00')
    assert result['last_source_at'].startswith('2026-09-04T18:42:00')
    assert 'verified_cbr_receipt_at=' in result['availability_evidence']


def test_runtime_gate_shifts_planned_market_update_after_late_receipt():
    moscow = ZoneInfo('Europe/Moscow')
    query = dt.datetime(2026, 9, 4, 19, 15, tzinfo=moscow)
    receipt = dt.datetime(2026, 9, 4, 19, 2, tzinfo=moscow)
    result = score_runtime_as_of(
        _receipt_gate_snapshots(), 'KZT', query, 5,
        verified_receipt_at=receipt)
    assert result['source_kind'] == 'post_receipt_market'
    assert result['probability_now_best_h'] == .82
    assert result['last_source_at'].startswith('2026-09-04T19:02:00')


@pytest.mark.parametrize('receipt', [
    '2026-09-04T19:00:00+03:00',
    '2026-09-03T18:30:00+03:00',
    '2026-09-04T18:30:00',
])
def test_runtime_gate_rejects_invalid_receipt_events(receipt):
    moscow = ZoneInfo('Europe/Moscow')
    query = dt.datetime(2026, 9, 4, 18, 45, tzinfo=moscow)
    with pytest.raises(ValueError, match='verified_receipt_at'):
        receipt_gated_snapshots(
            _receipt_gate_snapshots(), query,
            verified_receipt_at=receipt)


def test_runtime_gate_is_invariant_to_future_rows():
    moscow = ZoneInfo('Europe/Moscow')
    query = dt.datetime(2026, 9, 4, 18, 45, tzinfo=moscow)
    receipt = dt.datetime(2026, 9, 4, 18, 42, tzinfo=moscow)
    source = _receipt_gate_snapshots()
    first = score_runtime_as_of(
        source, 'KZT', query, 5, verified_receipt_at=receipt)
    changed = source.copy()
    future = changed.valid_from.str.startswith('2026-09-04T19:00')
    changed.loc[future, 'probability_h5'] = .001
    changed.loc[future, 'source_at'] = '2026-09-04T19:00:00+03:00'
    second = score_runtime_as_of(
        changed, 'KZT', query, 5, verified_receipt_at=receipt)
    assert first == second
