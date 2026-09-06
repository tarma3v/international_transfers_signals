import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t39_pre2025_currency_shrink import CANDIDATE
from research.temperature_t39_pre2025_currency_shrink_audit import audit


OUT = Path("results/research/temperature/t39_pre2025_currency_shrink")
T37_CANDIDATE = "source_driven_h20_shrink50_shadow"


def test_frozen_currency_map_uses_only_validated_pre2025_changes():
    frozen = pd.read_csv(OUT / "frozen_map.csv").set_index("currency")
    assert frozen.final_alpha.to_dict() == {
        "AMD": 1.0,
        "KGS": 0.5,
        "KZT": 0.5,
        "TJS": 0.5,
        "UZS": 1.0,
    }
    assert frozen.changed_from_t37.sum() == 2


def test_saved_candidate_is_bitwise_t37_outside_history():
    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    non_history = ~predictions.snapshot_source_kind.eq("cbr_history")
    assert np.array_equal(
        predictions.loc[non_history, CANDIDATE].to_numpy(),
        predictions.loc[non_history, T37_CANDIDATE].to_numpy(),
    )


def test_saved_t39_audit_rejects_repair_without_promotion():
    result = audit()
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert result["retrospective_repair_passed"] is False
    assert metadata["retrospective_repair_passed"] is False
    assert result["pre2025_selection_only"] is True
    assert result["future_prefix_corruption_passed"] is True
    assert result["production_promoted"] is False
