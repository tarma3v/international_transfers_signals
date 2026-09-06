import json
from pathlib import Path

import pandas as pd

from research.temperature_t27_rolling_origin_history_audit import audit


OUT = Path("results/research/temperature/t27_rolling_origin_history")


def test_saved_rolling_origin_fits_are_strictly_causal():
    log = pd.read_csv(OUT / "historical_fit_log.csv")
    assert len(log) == 4
    assert (log.latest_training_maturity_ord < log.cutoff_ord).all()
    assert (
        pd.to_datetime(log.latest_training_date) < pd.to_datetime(log.origin)
    ).all()


def test_saved_t27_audit_passes_and_discloses_open_period():
    result = audit()
    assert result["selected_model"] == "t25_base"
    assert result["passed"] is False
    assert result["fresh_independent_holdout"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["selection_on_open_period"] is False
    assert metadata["open_period_previously_inspected"] is True
