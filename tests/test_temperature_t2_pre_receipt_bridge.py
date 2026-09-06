import datetime as dt

import numpy as np

from research.temperature_t2_pre_receipt_bridge import post_window_delta


def _row(begin, end, close):
    return {'begin': begin, 'end': end, 'open': close, 'close': close,
            'high': close, 'low': close}


def test_post_window_delta_uses_only_completed_same_day_candles():
    day = dt.date(2024, 1, 8)
    history = {
        'CNYRUB_TOM': [
            _row(dt.datetime(2024, 1, 8, 15, 10),
                 dt.datetime(2024, 1, 8, 15, 19, 59), 10.),
            _row(dt.datetime(2024, 1, 8, 16, 0),
                 dt.datetime(2024, 1, 8, 16, 9, 59), 11.),
            _row(dt.datetime(2024, 1, 8, 16, 30),
                 dt.datetime(2024, 1, 8, 16, 39, 59), 50.),
        ]
    }
    index = [('TJS', 0, day)]
    result, available = post_window_delta(index, history, dt.time(16, 30))
    assert available[0]
    assert abs(result[0] - 10000. * np.log(1.1)) < 1e-9
