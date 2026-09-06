import datetime as dt

import numpy as np

from research.after_publication_ap32_effective_models import decision_deficit_router


def test_decision_router_prioritizes_core_and_caps_each_week():
    dates = np.array([dt.date(2025, 1, day) for day in range(1, 9)])
    currencies = np.array(['AMD'] * len(dates))
    core = np.array([False, True, True, True, False, False, False, False])
    fallback = np.ones(len(dates), dtype=bool)
    signal, _, reason = decision_deficit_router(
        core, fallback, dates, currencies, np.ones(len(dates), dtype=bool))
    assert signal.sum() <= 4
    assert reason[1] == 1
    assert set(reason).issubset({0, 1, 2})
