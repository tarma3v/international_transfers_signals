import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t31_mature_fixed_share import CANDIDATES, _online_candidates
from research.temperature_t31_mature_fixed_share_audit import audit


OUT = Path("results/research/temperature/t31_mature_fixed_share")


def _toy_frame():
    rows = []
    start = dt.date(2024, 1, 1)
    for day_offset in range(70):
        query = start + dt.timedelta(days=day_offset)
        for currency_offset, currency in enumerate(("AMD", "KZT")):
            phase = day_offset + currency_offset
            rows.append({
                "query_date": query,
                "currency": currency,
                "target": float((phase % 5) < 2),
                "maturity_ord": query.toordinal() + 20,
                "identity_early": 0.15 + 0.01 * (phase % 7),
                "all_platt_b050": 0.25 + 0.01 * (phase % 9),
                "recent2y_platt_b100": 0.35 + 0.01 * (phase % 11),
            })
    return pd.DataFrame(rows)


def test_future_targets_do_not_change_prefix_predictions():
    frame = _toy_frame()
    first, outputs_first, _, _ = _online_candidates(frame)
    changed = frame.copy()
    changed.loc[pd.to_datetime(changed.query_date).dt.date >= dt.date(2024, 2, 10), "target"] = (
        1.0 - changed.loc[
            pd.to_datetime(changed.query_date).dt.date >= dt.date(2024, 2, 10), "target"])
    second, outputs_second, _, _ = _online_candidates(changed)
    prefix = pd.to_datetime(first.query_date).dt.date <= dt.date(2024, 2, 28)
    assert first.loc[prefix, ["query_date", "currency"]].equals(
        second.loc[prefix, ["query_date", "currency"]])
    for name in CANDIDATES:
        assert np.allclose(outputs_first[name][prefix], outputs_second[name][prefix])


def test_online_state_is_normalized_and_uses_no_same_day_label():
    _, _, states, feedback = _online_candidates(_toy_frame())
    weights = states[["weight_identity", "weight_all", "weight_recent2y"]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0).all().all()
    used = states[states.latest_feedback_maturity_ord.notna()].copy()
    assert (pd.to_datetime(used.latest_feedback_publication_date)
            < pd.to_datetime(used.query_date)).all()
    assert (used.latest_feedback_maturity_ord < used.cutoff_ord).all()
    assert feedback["consumed_by_last_query"] < feedback["feedback_batches"]


def test_saved_t31_audit_passes():
    result = audit()
    assert result["screen_selected"] == "identity_early"
    assert result["validation_passed"] is False
    assert result["selected_model"] == "identity_early"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_2025_2026"] is False
    assert metadata["fresh_independent_holdout"] is False
