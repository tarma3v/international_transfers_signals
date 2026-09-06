import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t42_ood_history_shrink_audit import audit


OUT = Path("results/research/temperature/t42_ood_history_shrink")


def test_t42_distance_and_alpha_are_positive_and_bounded():
    predictions = pd.read_csv(OUT / "publication_predictions.csv.gz")
    assert predictions.ood_energy.gt(0.0).all()
    assert predictions.ood_alpha.between(0.0, 1.0).all()
    expected = np.minimum(1.0, 1.0 / predictions.ood_energy)
    np.testing.assert_allclose(predictions.ood_alpha, expected, atol=1e-12)


def test_t42_historical_gate_controls_open_and_never_promotes():
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["open_evaluated"] == metadata["historical_gate_passed"]
    assert metadata["production_promoted"] is False


def test_t42_packet_reconstructs_and_future_prefix_is_invariant():
    result = audit()
    assert result["source_hashes_match"] is True
    assert result["future_prefix_corruption_passed"] is True
    assert result["production_promoted"] is False
