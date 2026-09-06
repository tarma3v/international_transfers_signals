import datetime as dt

from research.temperature_t12_unified_router import (
    _early_source_timestamp,
    _stable_pre_receipt_bps,
)


def test_early_source_is_moscow_aware_and_stable_policy_is_fixed():
    assert _early_source_timestamp(
        dt.datetime(2026, 9, 4, 9, 59, 59)).endswith("+03:00")
    arrays = {
        "adaptive__cutoff_1530__h5": [12.],
        "prior__cutoff_1530__h5": [3.],
        "adaptive__cutoff_1530__h20": [99.],
        "prior__cutoff_1530__h20": [4.],
        "adaptive__premarket__h5": [88.],
        "prior__premarket__h5": [5.],
    }
    assert _stable_pre_receipt_bps(
        arrays, "cutoff_1530", 5, 0) == (12., "causal_adaptive")
    assert _stable_pre_receipt_bps(
        arrays, "cutoff_1530", 20, 0) == (4., "mature_prior")
    assert _stable_pre_receipt_bps(
        arrays, "premarket", 5, 0) == (5., "mature_prior")
