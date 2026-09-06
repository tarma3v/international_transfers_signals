import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t21_h20_curve_head import NUMERIC, _fit_candidates
from research.temperature_t21_h20_curve_head_audit import audit


OUT = Path("results/research/temperature/t21_h20_curve_head")


def _toy_frame(n, offset=0):
    index = np.arange(n) + offset
    frame = pd.DataFrame({
        column: np.sin(index / (7.0 + i)) + index / (1000.0 + 10 * i)
        for i, column in enumerate(NUMERIC)
    })
    frame["p20"] = np.clip(.15 + .7 * ((index % 17) / 16), .01, .99)
    frame["identity_h20"] = frame.p20
    frame["target"] = ((index * 7 + index // 11) % 13 < 4).astype(int)
    frame["currency"] = np.resize(["AMD", "KGS", "KZT", "TJS", "UZS"], n)
    frame["regime"] = np.resize(["market|spot|normal|fresh", "fallback"], n)
    return frame


def test_candidates_do_not_use_evaluation_labels():
    base = _toy_frame(600)
    calibration = _toy_frame(250, 1000)
    evaluation = _toy_frame(60, 2000)
    changed = evaluation.copy()
    changed["target"] = 1 - changed.target
    first, _ = _fit_candidates(base, calibration, evaluation)
    second, _ = _fit_candidates(base, calibration, changed)
    for model in first:
        assert np.allclose(first[model], second[model])


def test_saved_t21_audit_passes():
    result = audit()
    assert result["passing_states"] == 0
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
    assert metadata["checks"]["all_base_labels_mature"]
    assert metadata["checks"]["all_calibration_labels_mature"]
