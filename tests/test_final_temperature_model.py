import datetime as dt

from ml.final_temperature_model import (
    load_frozen_snapshots,
    score_final_all_horizons_as_of,
    score_final_table_as_of,
    verify_manifest,
)
from ml.transfer_temperature import MOSCOW


def test_final_manifest_hashes_and_statuses_are_frozen():
    manifest = verify_manifest()
    assert manifest["model_version"] == "final-temperature-v1-2026-09-07"
    assert "direct_pair_h5_bps" in manifest["shadow_components"]
    assert "direct_pair_probability" in manifest["rejected_components"]


def test_final_output_contains_all_horizons_and_never_enables_shadows():
    snapshots = load_frozen_snapshots()
    query = dt.datetime(2024, 1, 13, 10, 31, tzinfo=MOSCOW)
    result = score_final_all_horizons_as_of(
        "KZT", query, snapshots=snapshots
    )
    assert set(result["horizon_outputs"]) == {"1", "3", "5", "10", "20"}
    assert result["model_status"] == "frozen_hackathon_candidate"
    assert result["shadow_components_active"] is False
    for value in result["horizon_outputs"].values():
        assert 0 <= value["temperature_0_100"] <= 100
        assert value["last_source_at"] <= result["score_as_of"]


def test_final_receipt_gate_uses_event_not_fixed_clock():
    snapshots = load_frozen_snapshots()
    # The artifact begins with the 2024-01-09 evening receipt.  Use the next
    # business day so the no-receipt branch has a genuine pre-receipt snapshot.
    query = dt.datetime(2024, 1, 10, 18, 45, tzinfo=MOSCOW)
    before = score_final_all_horizons_as_of(
        "KZT", query, snapshots=snapshots
    )
    after = score_final_all_horizons_as_of(
        "KZT", query,
        verified_receipt_at=dt.datetime(2024, 1, 10, 18, 42, tzinfo=MOSCOW),
        snapshots=snapshots,
    )
    assert before["receipt_verified"] is False
    assert after["receipt_verified"] is True
    assert before["source_kind"] != "cbr_receipt"
    assert after["source_kind"] == "cbr_receipt"


def test_final_table_has_all_five_corridors():
    snapshots = load_frozen_snapshots()
    query = dt.datetime(2025, 5, 15, 15, 45, tzinfo=MOSCOW)
    table = score_final_table_as_of(query, snapshots=snapshots)
    assert {row["corridor"] for row in table} == {"AMD", "KGS", "KZT", "TJS", "UZS"}
