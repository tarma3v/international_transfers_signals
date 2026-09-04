import numpy as np
import pandas as pd

from research.version_b_honest_audit import past_oof_mask


def _scores(last_score: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "corridor": ["AMD"] * 6,
            "fold": [0, 0, 0, 1, 1, 1],
            "date": pd.date_range("2024-01-01", periods=6),
            "score": [0.1, 0.2, 0.3, 0.25, 0.35, last_score],
        }
    )


def test_rolling_threshold_cannot_change_an_earlier_decision_from_future_score():
    original, original_scope = past_oof_mask(_scores(0.4), .5, "rolling", 3)
    changed, changed_scope = past_oof_mask(_scores(100.0), .5, "rolling", 3)

    assert np.array_equal(original_scope, changed_scope)
    assert np.array_equal(original[:5], changed[:5])


def test_first_fold_only_seeds_past_oof_threshold_history():
    active, scope = past_oof_mask(_scores(0.4), .5, "rolling", 3)

    assert not scope[:3].any()
    assert scope[3:].all()
    assert not active[:3].any()
