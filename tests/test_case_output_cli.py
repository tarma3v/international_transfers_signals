import pandas as pd
import pytest

from run_case_output import REQUIRED_COLUMNS, build_table


def _snapshots():
    currencies = ['AMD', 'KGS', 'KZT', 'TJS', 'UZS']
    return pd.DataFrame({
        'currency': currencies,
        'valid_from': ['2026-09-04T15:30:00+03:00'] * len(currencies),
        'source_at': ['2026-09-04T15:19:59+03:00'] * len(currencies),
        'source_kind': ['moex_prefix'] * len(currencies),
        'phase': ['market_prefix'] * len(currencies),
        'confidence': ['mature_history'] * len(currencies),
        'availability_evidence': ['observed_completed_candle'] * len(currencies),
        'probability_h5': [.61] * len(currencies),
        'expected_future_bps_h5': [20.] * len(currencies),
        'push_now': [False] * len(currencies),
    })


def test_build_table_writes_required_columns_first(tmp_path):
    source = tmp_path / 'snapshots.csv'
    _snapshots().to_csv(source, index=False)
    table = build_table(source, '2026-09-04T16:00:00+03:00', 5)
    assert tuple(table.columns[:len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS
    assert len(table) == 5


def test_build_table_rejects_timezone_free_query(tmp_path):
    source = tmp_path / 'snapshots.csv'
    _snapshots().to_csv(source, index=False)
    with pytest.raises(ValueError, match='timezone'):
        build_table(source, '2026-09-04T16:00:00', 5)
