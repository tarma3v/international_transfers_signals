import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t44_calibration_quality_gate import (
    CANDIDATE,
    MIN_HALF_ROWS,
)
from research.temperature_t44_calibration_quality_gate_audit import audit


OUT = Path("results/research/temperature/t44_calibration_quality_gate")


def test_t44_has_disjoint_supported_gate_and_complete_grid():
    predictions = pd.read_csv(OUT / "publication_predictions.csv.gz")
    fit_log = pd.read_csv(OUT / "annual_fit_log.csv")
    gate = pd.read_csv(OUT / "quality_gate_metrics.csv")
    assert set(predictions.year.unique()) == set(range(2019, 2027))
    assert predictions.groupby("year").currency.nunique().eq(5).all()
    assert not predictions[["publication_date", "currency"]].duplicated().any()
    assert fit_log.early_calibration_rows.ge(MIN_HALF_ROWS).all()
    assert fit_log.quality_gate_rows.ge(MIN_HALF_ROWS).all()
    assert gate.groupby("year").size().eq(6).all()


def test_t44_rejected_years_equal_prior_and_gate_controls_open():
    predictions = pd.read_csv(OUT / "publication_predictions.csv.gz")
    fit_log = pd.read_csv(OUT / "annual_fit_log.csv")
    metadata = json.loads((OUT / "metadata.json").read_text())
    for year in fit_log.loc[~fit_log.gate_open, "year"]:
        part = predictions[predictions.year.eq(year)]
        assert np.array_equal(
            part[CANDIDATE].to_numpy(), part.causal_prior.to_numpy())
    assert metadata["open_evaluated"] == metadata["historical_gate_passed"]
    assert metadata["production_promoted"] is False


def test_t44_packet_reconstructs_and_passes_future_corruption():
    result = audit()
    assert result["source_hashes_match"] is True
    assert result["quality_gates_rebuilt"] is True
    assert result["future_prefix_corruption_passed"] is True
    assert result["production_promoted"] is False
