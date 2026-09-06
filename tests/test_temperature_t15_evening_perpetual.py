from pathlib import Path

import pandas as pd

from research.temperature_t15_evening_perpetual import OUT


def test_t15_keeps_probability_and_routes_only_screen_stable_benefit():
    selection = pd.read_csv(OUT / "selection.csv")
    probability = selection[selection.kind.eq("probability")]
    assert not probability.adopted.any()
    benefit = selection[
        selection.kind.eq("benefit") & selection.adopted]
    assert set(zip(benefit.clock, benefit.h)) == {
        ("perp_2000", 3), ("perp_2100", 3), ("perp_2100", 5),
        ("perp_2200", 3), ("perp_2200", 5),
        ("perp_2300", 3), ("perp_2300", 5),
    }
    assert (benefit.selected == "perp_dual_ridge").all()
    assert Path(OUT / "audit_checks.json").exists()
