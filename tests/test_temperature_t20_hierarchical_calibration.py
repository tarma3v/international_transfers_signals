import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t20_hierarchical_calibration import _fit_candidate
from research.temperature_t20_hierarchical_calibration_audit import audit


OUT = Path("results/research/temperature/t20_hierarchical_calibration")


def _toy_frame(labels):
    n = len(labels)
    return pd.DataFrame({
        "probability": np.linspace(.1, .9, n),
        "target": labels,
        "currency": np.resize(["AMD", "KGS", "KZT", "TJS", "UZS"], n),
        "phase": "market",
        "source_kind": "moex_prefix",
        "confidence": "normal",
        "freshness": "fresh",
    })


def test_candidate_does_not_use_evaluation_labels():
    train = _toy_frame(np.resize([0, 1], 300))
    evaluation = _toy_frame(np.resize([0, 1], 40))
    changed = evaluation.copy()
    changed["target"] = 1 - changed.target
    first, _ = _fit_candidate(train, evaluation, "hierarchical_beta")
    second, _ = _fit_candidate(train, changed, "hierarchical_beta")
    assert np.allclose(first, second)


def test_saved_t20_audit_passes():
    result = audit()
    assert result["passing_states"] == 0
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
    assert metadata["checks"]["all_training_labels_mature_before_cutoff"]
