import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import build_targets
from research.round5_features import load_round5_features
from research.temperature_t24_history_h20_anchor import _maturity
from research.temperature_t32_oos_warm_hedge import (
    WARM_START,
    _fit_q4_identity_anchor,
)
from research.temperature_t32_oos_warm_hedge_audit import audit


OUT = Path("results/research/temperature/t32_oos_warm_hedge")


def test_future_targets_do_not_change_q4_anchor():
    X, names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    target = build_targets(series, index)["fav_h20"]
    maturity = _maturity(series, index)
    first, first_log = _fit_q4_identity_anchor(
        X, names, dates, target, maturity)
    changed = target.copy()
    changed[dates >= WARM_START] = 1.0 - changed[dates >= WARM_START]
    second, second_log = _fit_q4_identity_anchor(
        X, names, dates, changed, maturity)
    assert np.allclose(first, second, equal_nan=True)
    assert first_log == second_log


def test_screen_really_starts_after_warm_feedback():
    states = pd.read_csv(OUT / "states.csv.gz")
    screen = states[pd.to_datetime(states.query_date).dt.year.eq(2023)]
    assert screen.consumed_feedback_batches.min() > 0
    assert (pd.to_datetime(screen.latest_feedback_publication_date)
            < pd.to_datetime(screen.query_date)).all()


def test_saved_t32_audit_passes():
    result = audit()
    assert result["screen_selected"] == "identity_early"
    assert result["validation_passed"] is False
    assert result["selected_model"] == "identity_early"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_2025_2026"] is False
    assert metadata["fresh_independent_holdout"] is False
