import datetime as dt

import numpy as np

from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)


def test_calibrator_respects_fixed_minimum_train_date():
    dates = np.array([
        dt.date(2021, 1, 1) + dt.timedelta(days=i * 7)
        for i in range(130)
    ], dtype=object)
    maturity = np.array([day + dt.timedelta(days=1) for day in dates],
                        dtype=object)
    raw = np.linspace(.05, .95, len(dates))
    target = (raw > .5).astype(float)
    currencies = np.array(['CNY'] * len(dates))
    _p, _prior, n_train, logs = fit_quarterly_calibrator(
        raw, target, maturity, dates, currencies,
        min_train_date=dt.date(2022, 2, 24))
    for row in logs:
        assert row['min_train_date'] == '2022-02-24'
    first_2023 = np.flatnonzero(dates >= dt.date(2023, 1, 1))[0]
    expected = sum(
        dt.date(2022, 2, 24) <= day < dt.date(2022, 12, 30)
        for day in dates
    )
    assert n_train[first_2023] == expected
