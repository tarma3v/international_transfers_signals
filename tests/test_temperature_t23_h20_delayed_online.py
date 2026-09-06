import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t23_h20_delayed_online import _monthly_fit
from research.temperature_t23_h20_delayed_online_audit import audit


OUT = Path("results/research/temperature/t23_h20_delayed_online")


def _mapper_frame(n, offset=0):
    index = np.arange(n) + offset
    return pd.DataFrame({
        "query_date": pd.date_range("2023-01-01", periods=n, freq="D").date,
        "currency": np.resize(["AMD", "KGS", "KZT", "TJS", "UZS"], n),
        "identity_h20": np.clip(.08 + .8 * ((index % 23) / 22), .01, .99),
        "frozen_logit": np.linspace(-2.5, 1.2, n),
        "residual_z": np.sin(index / 9.0),
        "target": ((index * 7 + index // 5) % 19 < 5).astype(int),
    })


def test_monthly_fit_does_not_use_prediction_targets():
    history = _mapper_frame(1000)
    evaluation = _mapper_frame(75, 2000)
    changed = evaluation.copy()
    changed["target"] = 1 - changed.target
    first, first_details = _monthly_fit(history, evaluation)
    second, second_details = _monthly_fit(history, changed)
    assert first_details["selected_family"] == second_details["selected_family"]
    for model in first:
        assert np.allclose(first[model], second[model])


def test_saved_t23_audit_passes():
    result = audit()
    assert result["passing_states"] == 0
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
    assert metadata["checks"]["all_monthly_labels_mature_before_cutoff"]
