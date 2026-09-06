import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t35_unified_h20_phase_router import (
    CANDIDATE,
    _build_route,
)
from research.temperature_t35_unified_h20_phase_router_audit import audit


OUT = Path("results/research/temperature/t35_unified_h20_phase_router")


def _toy_inputs():
    rows = []
    for scenario in ("calendar_assumed_replay", "no_same_day_receipt"):
        for clock in ("09:15", "12:45", "18:45"):
            rows.append({
                "scenario": scenario,
                "query_date": pd.Timestamp("2025-01-10").date(),
                "clock": clock,
                "currency": "AMD",
                "target": 1.0,
                "identity_h20": 0.40,
                "rank_correction_selected": 0.70,
            })
    t22 = pd.DataFrame(rows)
    t34 = pd.DataFrame({
        "query_date": [pd.Timestamp("2025-01-09").date()],
        "currency": ["AMD"],
        "target": [1.0],
        "cold_identity_qstack_w125_r100": [0.60],
    })
    return t22, t34


def test_phase_route_uses_only_registered_information_states():
    t22, t34 = _toy_inputs()
    output = _build_route(t22, t34)
    for row in output.itertuples():
        if row.clock == "09:15":
            assert row.route_source == "t34_early_publication"
            assert np.isclose(getattr(row, CANDIDATE), 0.60)
        elif row.clock == "18:45" and row.scenario == "calendar_assumed_replay":
            assert row.route_source == "t22_after_receipt_replay"
            assert np.isclose(getattr(row, CANDIDATE), 0.70)
        else:
            assert row.route_source == "t19_market_or_hold"
            assert np.isclose(getattr(row, CANDIDATE), 0.40)


def test_future_t34_rows_do_not_change_earlier_route():
    t22, t34 = _toy_inputs()
    first = _build_route(t22, t34)
    future = pd.DataFrame({
        "query_date": [pd.Timestamp("2025-02-01").date()],
        "currency": ["AMD"],
        "target": [0.0],
        "cold_identity_qstack_w125_r100": [0.99],
    })
    second = _build_route(t22, pd.concat([t34, future], ignore_index=True))
    assert np.allclose(first[CANDIDATE], second[CANDIDATE], atol=0.0)


def test_saved_t35_audit_passes_without_production_promotion():
    result = audit()
    assert result["retrospective_route_passed"] is False
    assert result["production_promoted"] is False
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["production_requires_verified_receipt_at"] is True
    assert metadata["changes_push_policy"] is False
