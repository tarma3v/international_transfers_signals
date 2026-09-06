import json
from pathlib import Path

import pandas as pd

from research.temperature_t43_nonlinear_history import MODEL_PARAMS
from research.temperature_t43_nonlinear_history_audit import audit


OUT = Path("results/research/temperature/t43_nonlinear_history")


def test_t43_has_fixed_model_and_complete_annual_grid():
    predictions = pd.read_csv(OUT / "publication_predictions.csv.gz")
    fit_log = pd.read_csv(OUT / "annual_fit_log.csv")
    assert set(predictions.year.unique()) == set(range(2019, 2027))
    assert predictions.groupby("year").currency.nunique().eq(5).all()
    assert not predictions[["publication_date", "currency"]].duplicated().any()
    assert fit_log.feature_count.eq(41).all()
    assert fit_log.n_iter.eq(MODEL_PARAMS["max_iter"]).all()


def test_t43_gate_controls_open_evaluation_and_never_promotes():
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["open_evaluated"] == metadata["historical_gate_passed"]
    assert metadata["production_promoted"] is False


def test_t43_saved_packet_reconstructs_and_passes_future_corruption():
    result = audit()
    assert result["source_hashes_match"] is True
    assert result["future_prefix_corruption_passed"] is True
    assert result["production_promoted"] is False
