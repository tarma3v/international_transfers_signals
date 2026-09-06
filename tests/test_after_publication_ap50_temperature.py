import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_as_of
from research.after_publication_ap50_temperature_models import (
    expected_calibration_error,
)


def test_latest_valid_temperature_never_reads_future_snapshot():
    moscow = ZoneInfo('Europe/Moscow')
    panel = pd.DataFrame({
        'currency': ['TJS', 'TJS'],
        'decision_at': ['2024-01-08T18:00:00+03:00',
                        '2024-01-09T18:00:00+03:00'],
        'availability_evidence': ['calendar_assumed', 'calendar_assumed'],
    })
    arrays = {
        'calibrated_probability_5': np.array([.7, .1]),
        'head_probability_5': np.array([.8, .2]),
        'composite_calibrated_probability': np.array([.6, .2]),
        'calibration_n_train_5': np.array([700, 700]),
        'signal__push': np.array([True, False]),
    }
    result = score_as_of(
        panel, arrays, 'TJS',
        dt.datetime(2024, 1, 9, 11, tzinfo=moscow), 5, 'push')
    assert result['probability_now_best_h'] == .7
    assert result['phase'] == 'before_new_cbr'
    assert result['push_now'] is True


def test_temperature_requires_timezone_and_marks_old_weekend_score_stale():
    panel = pd.DataFrame({
        'currency': ['TJS'],
        'decision_at': ['2024-01-05T18:00:00+03:00'],
        'availability_evidence': ['calendar_assumed'],
    })
    arrays = {
        'calibrated_probability_5': np.array([.4]),
        'head_probability_5': np.array([.5]),
        'composite_calibrated_probability': np.array([.45]),
        'calibration_n_train_5': np.array([300]),
    }
    try:
        score_as_of(panel, arrays, 'TJS', dt.datetime(2024, 1, 6), 5)
        assert False, 'naive timestamp accepted'
    except ValueError:
        pass
    result = score_as_of(
        panel, arrays, 'TJS',
        dt.datetime(2024, 1, 7, 18, tzinfo=ZoneInfo('Europe/Moscow')), 5)
    assert result['freshness'] == 'stale'
    assert result['phase'] == 'weekend_or_holiday'
    assert result['confidence'] == 'limited'


def test_ece_is_zero_for_perfect_fixed_bins():
    assert expected_calibration_error(
        np.array([0., 0., 1., 1.]), np.array([0., 0., 1., 1.])) < 1e-5
