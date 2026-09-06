import json
from pathlib import Path

import pandas as pd

from research.temperature_t38_h20_local_stability import (
    LOCAL_SLICES,
    _failure_summary,
)
from research.temperature_t38_h20_local_stability_audit import audit


OUT = Path("results/research/temperature/t38_h20_local_stability")


def test_failure_summary_counts_each_frozen_slice():
    clock = pd.DataFrame({
        "slice": ["currency", "currency", "year", "currency_year"],
        "noninferior": [True, False, True, False],
        "ece_delta_identity": [0.0, 0.02, 0.0, 0.0],
        "auc_delta_identity": [0.0, 0.0, 0.0, -0.01],
    })
    pooled = pd.DataFrame({
        "slice": ["currency", "year", "currency_year"],
        "pass": [True, False, False],
    })
    result = _failure_summary(clock, pooled).set_index("slice")
    assert set(result.index) == set(LOCAL_SLICES)
    assert result.loc["currency", "clock_ece_failures"] == 1
    assert result.loc["currency_year", "clock_auc_failures"] == 1
    assert result.loc["year", "pooled_failures"] == 1


def test_saved_t38_audit_matches_nonproduction_metadata():
    result = audit()
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert result["local_stability_passed"] == metadata[
        "local_stability_passed"]
    assert result["production_promoted"] is False
    assert metadata["model_changed"] is False
    assert metadata["local_bootstrap_inspected_before_registration"] is False
