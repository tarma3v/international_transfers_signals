import numpy as np

from research.temperature_t3_phase_calibration import loadz
from research.temperature_t10_early1000 import OUT, SELECTED, T4, T6, T9, selected_keys


def test_t10_routes_market_only_when_available():
    result = loadz(OUT / "outputs.npz")
    t4, t6, t9 = (loadz(path / "outputs.npz") for path in (T4, T6, T9))
    available = result["available"].astype(bool)
    assert available.any() and (~available).any()
    for h in SELECTED:
        _raw, probability = selected_keys(h)
        np.testing.assert_array_equal(
            result[f"routed_probability_h{h}"][available],
            t9[probability][available])
        np.testing.assert_array_equal(
            result[f"routed_probability_h{h}"][~available],
            t4[f"prob__history_hist__h{h}"][~available])
        np.testing.assert_array_equal(
            result[f"routed_expected_bps_h{h}"][~available],
            t6[f"expected_bps__premarket__h{h}"][~available])
