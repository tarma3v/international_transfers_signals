import datetime as dt

import pandas as pd

from research.temperature_t7_after_receipt_market import market_delta


def _row(begin, end, close):
    return {'begin': begin, 'end': end, 'open': close, 'close': close,
            'high': close, 'low': close}


def test_after_receipt_delta_uses_only_completed_candles_before_clock():
    panel = pd.DataFrame({'date': [dt.date(2024, 1, 8)]})
    history = {'CNYRUB_TOM': [
        _row(dt.datetime(2024, 1, 8, 17, 40),
             dt.datetime(2024, 1, 8, 17, 49, 59), 10.),
        _row(dt.datetime(2024, 1, 8, 18, 0),
             dt.datetime(2024, 1, 8, 18, 9, 59), 11.),
        _row(dt.datetime(2024, 1, 8, 18, 30),
             dt.datetime(2024, 1, 8, 18, 39, 59), 50.),
    ]}
    value, available = market_delta(panel, history, dt.time(18, 30))
    assert available[0]
    assert abs(value[0] - 10000. * __import__('numpy').log(1.1)) < 1e-9
