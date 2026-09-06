import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t34_cold_start_identity_gate import (
    CANDIDATE,
    MIN_FEEDBACK_BATCHES,
    _apply_gate,
)
from research.temperature_t34_cold_start_identity_gate_audit import audit


OUT = Path("results/research/temperature/t34_cold_start_identity_gate")


def _toy():
    frame = pd.DataFrame({
        "query_date": [pd.Timestamp("2024-01-10").date(),
                       pd.Timestamp("2024-04-10").date()],
        "currency": ["AMD", "AMD"],
        "target": [0.0, 1.0],
        "identity_early": [0.40, 0.60],
        "qstack_w125_r100": [0.20, 0.80],
    })
    states = pd.DataFrame({
        "quarter_origin": [pd.Timestamp("2024-01-01").date(),
                           pd.Timestamp("2024-04-01").date()],
        "cutoff_ord": [738884, 738975],
        "eligible_feedback_batches": [0, MIN_FEEDBACK_BATCHES],
        "selected_feedback_batches": [0, MIN_FEEDBACK_BATCHES],
        "latest_feedback_publication_date": [None, "2024-02-29"],
        "latest_feedback_maturity_ord": [np.nan, 738974],
    })
    return frame, states


def test_gate_falls_back_only_before_minimum_feedback():
    frame, states = _toy()
    output = _apply_gate(frame, states)
    assert output.cold_start.tolist() == [True, False]
    assert np.allclose(output[CANDIDATE], [0.40, 0.80])


def test_current_predictions_ignore_target_values():
    frame, states = _toy()
    first = _apply_gate(frame, states)
    changed = frame.copy()
    changed["target"] = 1.0 - changed.target
    second = _apply_gate(changed, states)
    assert np.allclose(first[CANDIDATE], second[CANDIDATE], atol=0.0)


def test_saved_t34_audit_passes_without_production_promotion():
    result = audit()
    assert result["screen_passed"] is True
    assert result["validation_passed"] is True
    assert result["historical_protocol_passed"] is True
    assert result["production_promoted"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_2025_2026"] is False
    assert metadata["fresh_independent_holdout"] is False
