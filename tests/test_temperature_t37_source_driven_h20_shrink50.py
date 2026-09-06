import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t37_source_driven_h20_shrink50 import (
    CANDIDATE,
    _apply_shrink_route,
    _equal_logit_blend,
)
from research.temperature_t37_source_driven_h20_shrink50_audit import audit


OUT = Path("results/research/temperature/t37_source_driven_h20_shrink50")


def _toy():
    return pd.DataFrame({
        "scenario": ["no_same_day_receipt", "no_same_day_receipt",
                     "calendar_assumed_replay"],
        "snapshot_source_kind": ["cbr_history", "moex_prefix", "cbr_receipt"],
        "identity_h20": [0.4, 0.5, 0.6],
        "t34_probability": [0.7, 0.8, 0.9],
        "rank_correction_selected": [0.2, 0.3, 0.75],
        "target": [0.0, 1.0, 1.0],
    })


def test_fixed_half_logit_shrink_only_changes_history_route():
    output = _apply_shrink_route(_toy())
    expected = _equal_logit_blend([0.4], [0.7])[0]
    assert output.route_source.tolist() == [
        "t34_cbr_history_shrink50", "t19_market_bridge_or_hold",
        "t22_after_receipt_replay"]
    assert np.allclose(output[CANDIDATE], [expected, 0.5, 0.75])


def test_target_values_do_not_change_shrunk_route():
    first = _apply_shrink_route(_toy())
    changed = _toy()
    changed["target"] = 1.0 - changed.target
    second = _apply_shrink_route(changed)
    assert np.allclose(first[CANDIDATE], second[CANDIDATE], atol=0.0)
    assert first.route_source.equals(second.route_source)


def test_saved_t37_audit_matches_metadata_without_promotion():
    result = audit()
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert result["retrospective_route_passed"] == metadata[
        "retrospective_route_passed"]
    assert result["production_promoted"] is False
    assert metadata["shrink_weight"] == 0.5
    assert metadata["shrink_weight_grid_searched"] is False
