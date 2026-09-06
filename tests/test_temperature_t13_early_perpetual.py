from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t3_phase_calibration import loadz
from research.temperature_t13_early_perpetual import CANDIDATES, OUT


def test_t13_selection_and_exact_fallback_are_persisted():
    result = loadz(OUT / "outputs.npz")
    selection = pd.read_csv(OUT / "selection.csv").set_index(["clock", "h"])
    assert selection.loc[("perp_0900", 1), "selected"] == "perp_basis_rank"
    assert selection.loc[("perp_0900", 3), "selected"] == "perp_basis_rank"
    assert set(selection.loc["perp_1000", "selected"]) == {"control"}
    for clock in ("perp_0900", "perp_1000"):
        for h in (1, 3, 5, 10, 20):
            control = result[f"control__{clock}__h{h}"]
            for candidate in CANDIDATES:
                prefix = f"{clock}__{candidate}__h{h}"
                usable = result[f"usable__{prefix}"].astype(bool)
                routed = result[f"routed__{prefix}"]
                np.testing.assert_array_equal(routed[~usable], control[~usable])

    checks = Path(OUT / "audit_checks.json")
    assert checks.exists()
