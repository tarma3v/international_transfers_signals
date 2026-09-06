import json
from pathlib import Path

import pandas as pd

from research.temperature_t41_mature_quarterly_shrink_audit import audit


OUT = Path("results/research/temperature/t41_mature_quarterly_shrink")


def test_t41_quarter_states_are_causal_and_bounded():
    states = pd.read_csv(OUT / "states.csv")
    assert not states.quarter_origin.duplicated().any()
    assert states.alpha.between(0.0, 1.0).all()
    used = states.latest_feedback_maturity_ord.notna()
    assert (states.loc[used, "latest_feedback_maturity_ord"]
            < states.loc[used, "cutoff_ord"]).all()


def test_t41_historical_gate_controls_open_and_never_promotes():
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["open_evaluated"] == metadata["historical_gate_passed"]
    assert metadata["production_promoted"] is False


def test_t41_packet_reconstructs_and_future_prefix_is_invariant():
    result = audit()
    assert result["source_hashes_match"] is True
    assert result["future_prefix_corruption_passed"] is True
    assert result["production_promoted"] is False
