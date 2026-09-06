import json
from pathlib import Path

import pandas as pd

from research.temperature_t40_long_rolling_history import YEARS
from research.temperature_t40_long_rolling_history_audit import audit


OUT = Path("results/research/temperature/t40_long_rolling_history")


def test_t40_has_complete_annual_currency_grid_and_embargo():
    predictions = pd.read_csv(OUT / "publication_predictions.csv.gz")
    fit_log = pd.read_csv(OUT / "annual_fit_log.csv")
    assert set(predictions.year.unique()) == set(YEARS)
    assert predictions.groupby("year").currency.nunique().eq(5).all()
    assert not predictions[["publication_date", "currency"]].duplicated().any()
    assert (fit_log.latest_training_maturity_ord < fit_log.fit_cutoff_ord).all()
    assert (fit_log.latest_calibration_maturity_ord
            < fit_log.calibration_cutoff_ord).all()


def test_t40_gate_controls_open_evaluation_and_never_promotes():
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["open_evaluated"] == metadata["historical_gate_passed"]
    assert metadata["production_promoted"] is False


def test_t40_saved_packet_reconstructs_and_passes_future_corruption():
    result = audit()
    assert result["source_hashes_match"] is True
    assert result["future_prefix_corruption_passed"] is True
    assert result["production_promoted"] is False
