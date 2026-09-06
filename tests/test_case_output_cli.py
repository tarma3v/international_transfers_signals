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


def test_build_table_defaults_to_verified_receipt_gate(tmp_path):
    source = tmp_path / 'snapshots.csv'
    snapshots = _snapshots()
    receipt = snapshots.copy()
    receipt['valid_from'] = '2026-09-04T18:30:00+03:00'
    receipt['source_at'] = '2026-09-04T18:30:00+03:00'
    receipt['source_kind'] = 'cbr_receipt'
    receipt['phase'] = 'after_new_cbr'
    receipt['availability_evidence'] = 'calendar_assumed'
    receipt['probability_h5'] = .91
    pd.concat([snapshots, receipt], ignore_index=True).to_csv(source, index=False)

    held = build_table(source, '2026-09-04T18:45:00+03:00', 5)
    assert held.source_kind.eq('moex_prefix').all()
    verified = build_table(
        source, '2026-09-04T18:45:00+03:00', 5,
        verified_receipt_at='2026-09-04T18:42:00+03:00')
    assert verified.source_kind.eq('cbr_receipt').all()
    assert verified.receipt_verified.all()


def test_build_table_requires_one_receipt_mode(tmp_path):
    source = tmp_path / 'snapshots.csv'
    _snapshots().to_csv(source, index=False)
    with pytest.raises(ValueError, match='exclusive'):
        build_table(
            source, '2026-09-04T18:45:00+03:00', 5,
            verified_receipt_at='2026-09-04T18:42:00+03:00',
            historical_calendar_assumption=True)
