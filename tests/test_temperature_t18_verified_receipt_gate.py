import json

from research.temperature_t18_verified_receipt_gate import OUT, run_audit


def test_t18_verified_receipt_gate_audit():
    result = run_audit()
    checks = result["checks"]
    assert checks["all_five_corridors_before_receipt"]
    assert checks["no_same_day_receipt_model_without_event"]
    assert checks["verified_event_activates_cbr_row"]
    assert checks["late_receipt_delays_1900_market_update"]
    assert checks["numeric_outputs_and_push_preserved"]
    assert checks["future_row_corruption_prefix_invariant"]
    assert checks["historical_receipts_certified"] is False
    assert checks["bank_execution_validated"] is False


def test_t18_saved_metadata_keeps_research_limitations_explicit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["production_default_requires_verified_receipt"] is True
    assert metadata["historical_calendar_assumption_explicit_only"] is True
    assert metadata["predictive_outputs_refit"] is False
    assert metadata["selection_used_open_outcomes"] is False
    assert metadata["historical_receipts_certified"] is False
    assert metadata["bank_execution_validated"] is False
