import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t24_history_h20_anchor import (
    MODELS,
    _calibrate_and_select,
)
from research.temperature_t24_history_h20_anchor_audit import audit


OUT = Path("results/research/temperature/t24_history_h20_anchor")


def _toy_query():
    dates = pd.date_range("2024-01-01", "2026-08-05", freq="D")
    frame = pd.DataFrame({
        "query_date": np.repeat(dates.date, 5),
        "currency": np.tile(["AMD", "KGS", "KZT", "TJS", "UZS"], len(dates)),
    })
    index = np.arange(len(frame))
    frame["target"] = ((index * 7 + index // 13) % 23 < 6).astype(int)
    frame["identity_early"] = np.clip(.1 + .7 * ((index % 29) / 28), .01, .99)
    # The artificial maturity stays before both historical cutoffs.
    frame["maturity_ord"] = np.asarray([
        pd.Timestamp(day).date().toordinal() - 3 for day in frame.query_date
    ])
    raw = {
        name: np.clip(.08 + .8 * ((index * (i + 3)) % 31) / 30, .01, .99)
        for i, name in enumerate(MODELS[1:])
    }
    return frame, raw


def test_selection_and_predictions_ignore_evaluation_targets():
    query, raw = _toy_query()
    changed = query.copy()
    changed.loc[pd.to_datetime(changed.query_date).dt.year >= 2025, "target"] ^= 1
    first = _calibrate_and_select(query, raw)
    second = _calibrate_and_select(changed, raw)
    assert first[1] == second[1]
    for model in first[0]:
        assert np.allclose(first[0][model], second[0][model])


def test_saved_t24_audit_passes():
    result = audit()
    assert result["selected_model"] == "identity_early"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
