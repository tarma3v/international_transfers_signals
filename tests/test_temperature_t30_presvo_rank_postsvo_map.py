import datetime as dt
import json
from pathlib import Path

import numpy as np

from research.temperature_t30_presvo_rank_postsvo_map import (
    CANDIDATES,
    _build_candidates,
    _fit_positive_platt,
)
from research.temperature_t30_presvo_rank_postsvo_map_audit import audit


OUT = Path("results/research/temperature/t30_presvo_rank_postsvo_map")


def test_positive_platt_is_monotone():
    raw = np.linspace(.01, .99, 200)
    target = (raw > .58).astype(int)
    intercept, slope, _ = _fit_positive_platt(raw, target)
    mapped = 1.0 / (1.0 + np.exp(-(intercept + slope * np.log(raw / (1 - raw)))))
    assert slope > 0
    assert np.diff(mapped).min() >= 0


def test_future_targets_do_not_change_frozen_2022_maps():
    dates = np.asarray([
        dt.date(2022, 4, 1) + dt.timedelta(days=i) for i in range(245)
    ] + [
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(50)
    ], dtype=object)
    row = np.arange(len(dates))
    raw_models = {
        "all": np.clip(.05 + .8 * ((row * 7) % 31) / 30, .01, .99),
        "recent4y": np.clip(.06 + .75 * ((row * 11) % 37) / 36, .01, .99),
        "recent2y": np.clip(.07 + .7 * ((row * 13) % 41) / 40, .01, .99),
    }
    target = ((row * 5) % 17 < 6).astype(float)
    maturity = np.asarray([day.toordinal() + 20 for day in dates], dtype=float)
    first, first_log, _ = _build_candidates(raw_models, dates, target, maturity)
    changed = target.copy()
    changed[dates >= dt.date(2025, 1, 1)] = 1 - changed[dates >= dt.date(2025, 1, 1)]
    second, second_log, _ = _build_candidates(raw_models, dates, changed, maturity)
    for name in CANDIDATES:
        assert np.allclose(first[name], second[name])
    assert np.allclose(first_log[["intercept", "slope"]], second_log[["intercept", "slope"]])


def test_saved_t30_audit_passes():
    result = audit()
    assert result["screen_selected"] == "identity_early"
    assert result["validation_passed"] is False
    assert result["selected_model"] == "identity_early"
    assert result["passed"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_2025_2026"] is False
    assert metadata["fresh_independent_holdout"] is False
