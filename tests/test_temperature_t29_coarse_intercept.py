import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from research.temperature_t29_coarse_intercept import _coarse_candidates
from research.temperature_t29_coarse_intercept_audit import audit


OUT = Path("results/research/temperature/t29_coarse_intercept")


def _toy_frame():
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    frame = pd.DataFrame({
        "query_date": dates.date,
        "publication_date": dates.date,
        "currency": ["USD"] * len(dates),
        "target": ((np.arange(len(dates)) * 7) % 19 < 5).astype(int),
        "maturity_ord": [d.date().toordinal() + 20 for d in dates],
    })
    base = np.clip(.12 + .45 * ((np.arange(len(dates)) * 11) % 23) / 22, .01, .99)
    return frame, base


def test_coarse_period_ignores_targets_not_mature_at_boundary():
    frame, base = _toy_frame()
    first, _, _ = _coarse_candidates(frame, base)
    changed = frame.copy()
    april = changed.query_date >= dt.date(2024, 4, 10)
    changed.loc[april, "target"] ^= 1
    second, _, _ = _coarse_candidates(changed, base)
    may = ((changed.query_date >= dt.date(2024, 5, 1))
           & (changed.query_date < dt.date(2024, 6, 1)))
    for name in first:
        assert np.allclose(first[name][may], second[name][may])


def test_coarse_logit_delta_is_constant_inside_month():
    frame, base = _toy_frame()
    candidates, _, _ = _coarse_candidates(frame, base)
    month = pd.to_datetime(frame.query_date).dt.to_period("M")
    for name, values in candidates.items():
        delta = pd.Series(logit(values) - logit(base))
        assert delta.groupby(month).apply(lambda x: x.max() - x.min()).max() < 1e-10


def test_saved_t29_audit_passes():
    result = audit()
    assert result["screen_selected"] == "month_w30_b100"
    assert result["validation_passed"] is False
    assert result["selected_model"] == "t25_base"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_2025_2026"] is False
    assert metadata["fresh_independent_holdout"] is False
