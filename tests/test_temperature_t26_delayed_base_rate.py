import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t26_delayed_base_rate import (
    _delayed_candidates,
    _selection_metrics,
)
from research.temperature_t26_delayed_base_rate_audit import audit


OUT = Path("results/research/temperature/t26_delayed_base_rate")


def _toy_frame():
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    currencies = np.array(["AMD", "KGS", "KZT", "TJS", "UZS"])
    row = np.arange(len(dates) * len(currencies))
    date_index = np.repeat(np.arange(len(dates)), len(currencies))
    query_date = np.repeat(dates.date, len(currencies))
    frame = pd.DataFrame({
        "query_date": query_date,
        "publication_date": query_date,
        "currency": np.tile(currencies, len(dates)),
        "target": ((row * 7 + date_index * 3) % 29 < 7).astype(int),
        "maturity_ord": np.repeat(
            [(day + pd.Timedelta(days=20)).date().toordinal() for day in dates],
            len(currencies),
        ),
    })
    base = np.clip(.08 + .32 * ((row * 11) % 31) / 30, .01, .99)
    return frame, base


def test_unmatured_target_corruption_leaves_prefix_unchanged():
    frame, base = _toy_frame()
    origin = pd.Timestamp("2024-02-01").date()
    cutoff = (pd.Timestamp(origin) - pd.Timedelta(days=2)).date().toordinal()
    changed = frame.copy()
    changed.loc[changed.maturity_ord >= cutoff, "target"] ^= 1
    first, _, _ = _delayed_candidates(frame, base, return_states=False)
    second, _, _ = _delayed_candidates(changed, base, return_states=False)
    prefix = pd.to_datetime(frame.query_date).dt.date <= origin
    for name in first:
        assert np.allclose(first[name][prefix], second[name][prefix])


def test_selection_ignores_later_evaluation_targets():
    frame, base = _toy_frame()
    candidates_first, _, _ = _delayed_candidates(frame, base, return_states=False)
    frame["t25_base"] = base
    dates = pd.to_datetime(frame.query_date)
    selection = ((dates >= "2023-07-01") & (dates < "2023-11-01")).to_numpy()
    evaluation = (dates >= "2024-01-01").to_numpy()
    selected_first, screen_first = _selection_metrics(frame, candidates_first, selection)

    changed = frame.copy()
    changed.loc[evaluation, "target"] ^= 1
    candidates_second, _, _ = _delayed_candidates(changed, base, return_states=False)
    selected_second, screen_second = _selection_metrics(changed, candidates_second, selection)
    assert selected_first == selected_second
    assert screen_first == screen_second
    for name in candidates_first:
        assert np.allclose(candidates_first[name][selection], candidates_second[name][selection])


def test_saved_t26_audit_passes():
    result = audit()
    assert result["selected_model"] == "t25_base"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
