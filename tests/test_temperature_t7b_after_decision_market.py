import datetime as dt

import numpy as np
import pandas as pd

from research.temperature_t7b_after_decision_market import delayed_market_delta


def _row(begin, end, close):
    return {'begin': begin, 'end': end, 'open': close, 'close': close,
            'high': close, 'low': close}


def test_delayed_delta_uses_only_candles_new_after_1830_information_set():
    panel = pd.DataFrame({'date': [dt.date(2024, 1, 8)]})
    history = {'CNYRUB_TOM': [
        _row(dt.datetime(2024, 1, 8, 18, 0),
             dt.datetime(2024, 1, 8, 18, 9, 59), 10.),
        _row(dt.datetime(2024, 1, 8, 18, 30),
             dt.datetime(2024, 1, 8, 18, 39, 59), 11.),
        _row(dt.datetime(2024, 1, 8, 18, 40),
             dt.datetime(2024, 1, 8, 18, 49, 59), 50.),
    ]}
    value, available = delayed_market_delta(panel, history, dt.time(19, 0))
    assert available[0]
    assert abs(value[0] - 10000. * np.log(1.1)) < 1e-9
