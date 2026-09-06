import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t45_direct_pair_benefit_audit import audit


OUT = Path("results/research/temperature/t45_direct_pair_benefit")


def test_t45_historical_gates_open_only_preregistered_horizons():
    gates = pd.read_csv(OUT / "historical_gates.csv")
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert set(gates.period) == {"screen_2023", "validation_2024"}
    assert gates.groupby("period").h.nunique().eq(5).all()
    assert metadata["open_horizons"] == [3, 5, 10]
    assert gates[gates.h.isin([3, 5, 10])].stage_pass.all()
    assert not gates[gates.h.isin([1, 20])].groupby("h").stage_pass.all().any()
    assert metadata["production_promoted"] is False


def test_t45_saved_predictions_fallback_and_open_mask():
    predictions = pd.read_csv(OUT / "publication_predictions.csv.gz")
    historical = pd.to_datetime(predictions.date).dt.year.le(2024)
    ineligible = historical & ~predictions.eligible
    np.testing.assert_allclose(
        predictions.loc[ineligible, "candidate_expected_bps"],
        predictions.loc[ineligible, "baseline_expected_bps"],
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    opened = pd.to_datetime(predictions.date).dt.year.ge(2025)
    assert predictions.loc[opened & predictions.h.isin([1, 20]),
                           "candidate_expected_bps"].isna().all()
    assert predictions.loc[opened & predictions.h.isin([3, 5, 10]),
                           "candidate_expected_bps"].notna().any()


def test_t45_reconstructs_and_passes_future_corruption():
    checks = audit()
    assert checks["quarterly_predictions_rebuilt"] is True
    assert checks["future_prefix_corruption_passed"] is True
    assert checks["open_horizons"] == [3, 5, 10]
