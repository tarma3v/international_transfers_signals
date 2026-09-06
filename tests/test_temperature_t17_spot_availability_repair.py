import json

import pandas as pd

from research.temperature_t17_spot_availability_repair import OUT


def test_t17_artifact_uses_observed_spot_availability():
    metadata = json.loads((OUT / "metadata.json").read_text())
    checks = json.loads((OUT / "audit_checks.json").read_text())
    assert metadata["packet"] == "temperature-T17"
    assert metadata["dropped_synthetic_spot_rows"] > 0
    assert metadata["observed_spot_rows"] > 0
    assert metadata["snapshot_rows"] < metadata["base_rows"]
    assert checks["snapshot_artifact_exactly_rebuilt"] is True
    assert checks["kept_predictions_benefits_push_exact"] is True
    assert checks["future_spot_deletion_prefix_invariant"] is True


def test_t17_kept_spot_sources_precede_valid_from():
    frame = pd.read_csv(OUT / "snapshots.csv.gz")
    spot = frame.source_kind.isin({
        "moex_prefix", "post_window_market", "post_receipt_market",
    })
    source = pd.to_datetime(frame.loc[spot, "source_at"], utc=True)
    valid = pd.to_datetime(frame.loc[spot, "valid_from"], utc=True)
    assert (source < valid).all()
    assert frame.loc[spot, "availability_evidence"].eq(
        "observed_completed_same_day_cny_spot_strict_before_valid_from"
    ).all()
