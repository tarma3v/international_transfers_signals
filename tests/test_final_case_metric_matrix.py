from pathlib import Path

import pandas as pd


MATRIX = Path('results/research/final_case_metric_matrix.csv')


def test_final_matrix_covers_both_targets_all_horizons_and_corridors():
    frame = pd.read_csv(MATRIX)
    assert set(frame.target) == {'now_favourable', 'window_closing'}
    assert set(frame.h) == {1, 3, 5, 10, 20}
    assert set(frame.corridor) == {'AMD', 'KGS', 'KZT', 'TJS', 'UZS'}
    assert len(frame) == 300


def test_ap37_passes_point_gates_for_both_case_targets():
    frame = pd.read_csv(MATRIX)
    ap37 = frame[frame.indicator.eq('AP37 mature-precision router')]
    assert len(ap37) == 50
    assert ap37.lift.min() >= 1.3
    assert ap37.signals_per_week.min() >= 1.0
    assert ap37.signals_per_week.max() <= 2.0
    assert ap37.symmetric_bps.min() > 0.0
