import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t33_quarterly_stacking import (
    CANDIDATES,
    _quarterly_candidates,
)
from research.temperature_t33_quarterly_stacking_audit import audit


OUT = Path("results/research/temperature/t33_quarterly_stacking")


def _toy_frame():
    rows = []
    start = dt.date(2024, 1, 1)
    for offset in range(274):
        day = start + dt.timedelta(days=offset)
        for currency_offset, currency in enumerate(("AMD", "KGS", "KZT", "TJS", "UZS")):
            phase = offset + currency_offset
            rows.append({
                "query_date": day,
                "currency": currency,
                "target": float((phase % 7) < 2),
                "maturity_ord": day.toordinal() + 20,
                "identity_early": .15 + .01 * (phase % 7),
                "all_platt_b050": .25 + .01 * (phase % 9),
                "recent2y_platt_b100": .35 + .01 * (phase % 11),
            })
    return pd.DataFrame(rows)


def test_unavailable_targets_do_not_change_current_quarter():
    frame = _toy_frame()
    first, first_outputs, _ = _quarterly_candidates(frame)
    changed = frame.copy()
    change_mask = pd.to_datetime(changed.query_date).dt.date >= dt.date(2024, 3, 15)
    changed.loc[change_mask, "target"] = 1.0 - changed.loc[change_mask, "target"]
    second, second_outputs, _ = _quarterly_candidates(changed)
    prefix = pd.to_datetime(first.query_date).dt.date < dt.date(2024, 7, 1)
    assert first.loc[prefix, ["query_date", "currency"]].equals(
        second.loc[prefix, ["query_date", "currency"]])
    for name in CANDIDATES:
        assert np.allclose(first_outputs[name][prefix], second_outputs[name][prefix])


def test_weights_are_one_state_per_quarter_and_normalized():
    _, _, states = _quarterly_candidates(_toy_frame())
    assert not states[["quarter_origin", "candidate"]].duplicated().any()
    weights = states[["weight_identity", "weight_all", "weight_recent2y"]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0.0).all().all()
    used = states[states.latest_feedback_maturity_ord.notna()]
    assert (pd.to_datetime(used.latest_feedback_publication_date)
            < pd.to_datetime(used.quarter_origin)).all()
    assert (used.latest_feedback_maturity_ord < used.cutoff_ord).all()


def test_saved_t33_audit_passes():
    result = audit()
    assert result["screen_selected"] == "identity_early"
    assert result["validation_passed"] is False
    assert result["selected_model"] == "identity_early"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_2025_2026"] is False
    assert metadata["fresh_independent_holdout"] is False
