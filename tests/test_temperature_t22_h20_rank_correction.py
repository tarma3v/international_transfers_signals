import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t21_h20_curve_head import NUMERIC
from research.temperature_t22_h20_rank_correction import _fit_correction
from research.temperature_t22_h20_rank_correction_audit import audit


OUT = Path("results/research/temperature/t22_h20_rank_correction")


def _toy_frame(n, offset=0):
    index = np.arange(n) + offset
    frame = pd.DataFrame({
        column: np.cos(index / (5.0 + i)) + index / (900.0 + 10 * i)
        for i, column in enumerate(NUMERIC)
    })
    frame["p20"] = np.clip(.12 + .72 * ((index % 19) / 18), .01, .99)
    frame["identity_h20"] = frame.p20
    frame["target"] = ((index * 5 + index // 7) % 17 < 5).astype(int)
    frame["currency"] = np.resize(["AMD", "KGS", "KZT", "TJS", "UZS"], n)
    frame["regime"] = np.resize(["market|spot|normal|fresh", "fallback"], n)
    return frame


def test_correction_does_not_use_evaluation_labels():
    base = _toy_frame(600)
    calibration = _toy_frame(250, 1000)
    evaluation = _toy_frame(60, 2000)
    changed = evaluation.copy()
    changed["target"] = 1 - changed.target
    first, first_details = _fit_correction(base, calibration, evaluation)
    second, second_details = _fit_correction(base, calibration, changed)
    assert first_details["selected_alpha"] == second_details["selected_alpha"]
    for model in first:
        assert np.allclose(first[model], second[model])


def test_saved_t22_audit_passes():
    result = audit()
    assert result["passing_states"] == 6
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
    assert metadata["checks"]["base_only_residualization"]
