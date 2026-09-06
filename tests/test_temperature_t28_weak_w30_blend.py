import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t28_weak_w30_blend import (
    _blend_candidates,
    _candidate_metrics,
    _choose,
)
from research.temperature_t28_weak_w30_blend_audit import audit


OUT = Path("results/research/temperature/t28_weak_w30_blend")


def test_blend_selection_is_independent_of_later_targets():
    n = 360
    row = np.arange(n)
    frame = pd.DataFrame({
        "target": ((row * 7) % 23 < 6).astype(int),
        "t25_base": np.clip(.08 + .4 * ((row * 5) % 29) / 28, .01, .99),
    })
    w30 = np.clip(.06 + .5 * ((row * 11) % 31) / 30, .01, .99)
    candidates = _blend_candidates(frame.t25_base, w30)
    screen = row < 140
    validation = (row >= 140) & (row < 260)
    future = row >= 260
    first_screen = _candidate_metrics(frame, candidates, screen)
    first_validation = _candidate_metrics(frame, candidates, validation)

    changed = frame.copy()
    changed.loc[future, "target"] ^= 1
    second_screen = _candidate_metrics(changed, candidates, screen)
    second_validation = _candidate_metrics(changed, candidates, validation)
    assert first_screen == second_screen
    assert first_validation == second_validation
    assert _choose(first_screen) == _choose(second_screen)


def test_saved_t28_audit_passes():
    result = audit()
    assert result["screen_selected"] == "blend_w30_b050"
    assert result["validation_passed"] is False
    assert result["selected_model"] == "t25_base"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_2025_2026"] is False
    assert metadata["fresh_independent_holdout"] is False
