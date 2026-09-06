import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t25_anchor_preserving_map import (
    _build_maps,
    _select,
)
from research.temperature_t25_anchor_preserving_map_audit import audit


OUT = Path("results/research/temperature/t25_anchor_preserving_map")


def _toy_query():
    dates = pd.date_range("2024-01-01", periods=240, freq="D")
    currency = np.tile(["AMD", "KGS", "KZT", "TJS", "UZS"], len(dates))
    day = np.repeat(np.arange(len(dates)), 5)
    row = np.arange(len(currency))
    frame = pd.DataFrame({
        "query_date": np.repeat(dates.date, 5),
        "currency": currency,
        "target": ((row * 11 + day) % 19 < 5).astype(int),
        "identity_early": np.clip(.08 + .35 * ((row * 7) % 23) / 22, .01, .99),
    })
    compact = np.clip(.05 + .9 * ((row * 13 + day * 3) % 37) / 36, .01, .99)
    calibration = day < 100
    selection = (day >= 100) & (day < 180)
    evaluation = day >= 180
    return frame, compact, calibration, selection, evaluation


def test_maps_and_selection_ignore_evaluation_targets():
    query, compact, calibration, selection, evaluation = _toy_query()
    maps_first, details_first = _build_maps(query, compact, calibration)
    selected_first, screen_first = _select(query, maps_first, selection)

    changed = query.copy()
    changed.loc[evaluation, "target"] ^= 1
    maps_second, details_second = _build_maps(changed, compact, calibration)
    selected_second, screen_second = _select(changed, maps_second, selection)

    assert selected_first == selected_second
    assert details_first == details_second
    assert screen_first == screen_second
    for name in maps_first:
        assert np.allclose(maps_first[name], maps_second[name])


def test_saved_t25_audit_passes():
    result = audit()
    assert result["selected_model"] == "residual_a040"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
