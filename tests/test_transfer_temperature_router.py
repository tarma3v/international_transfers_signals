import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of


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
