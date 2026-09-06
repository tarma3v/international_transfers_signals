import datetime as dt

import numpy as np
import pandas as pd
import pytest
from ml.data import Series

from research.after_publication_panel import build_outcomes
from research.after_publication_reference_comparison import validate_references


def fixture(values):
    dates = np.array([dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(len(values))], dtype=object)
    s = Series('KZT', dates, np.array(values, dtype=float))
    p = pd.DataFrame([dict(currency='KZT', date=dates[1], current_index=1,
        announced_index=2, current_price=float(values[1]), announced_price=float(values[2]))])
    return {'KZT': s}, p


def test_effective_known_h1_can_reverse_publication_target():
    series, panel = fixture([99, 100, 90, 95, 96, 97, 98, 99])
    targets = {r: build_outcomes(series, panel, r) for r in ('effective', 'publication')}
    validate_references(series, panel, targets)
    assert targets['effective']['y5'][0] == 0
    assert targets['publication']['y5'][0] == 1


def test_known_rise_does_not_make_whole_h5_known():
    series, panel = fixture([99, 100, 110, 90, 95, 96, 97, 98])
    targets = {r: build_outcomes(series, panel, r) for r in ('effective', 'publication')}
    validate_references(series, panel, targets)
    assert targets['effective']['y1'][0] == 1
    assert targets['effective']['y5'][0] == 0


def test_equality_survives_and_endpoints_are_distinct():
    series, panel = fixture([99, 100, 100, 110, 110, 110, 110, 90])
    targets = {r: build_outcomes(series, panel, r) for r in ('effective', 'publication')}
    validate_references(series, panel, targets)
    assert targets['effective']['y5'][0] == 1
    assert targets['publication']['y5'][0] == 0


def test_reference_audit_rejects_wrong_current_index():
    series, panel = fixture([99, 100, 90, 95, 96, 97, 98, 99])
    targets = {r: build_outcomes(series, panel, r) for r in ('effective', 'publication')}
    panel.loc[0, 'current_index'] = 0
    with pytest.raises(AssertionError):
        validate_references(series, panel, targets)
