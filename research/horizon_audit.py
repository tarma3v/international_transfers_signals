"""Lock an anchor policy on 2022--2023 and audit every required horizon.

The score families and policy grid are fixed in code.  For each horizon, the
choice is made only on 2022--2023, then evaluated unchanged on 2024--2026.
The h=5 diagnostic champion is also reported explicitly as post-selection
analysis, rather than being confused with the locked result.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, benefit_forward_only, build_targets
from research.extended_features import load_or_build
from research.model_study import (
    FINAL_TEST_YEARS,
    REGIME_VALID_YEARS,
    TARGET_RATES,
    best_row,
    evaluate,
)

OUT = Path("results/research")
ANCHORS = ("anchor_pct90", "anchor_multiscale", "anchor_trend", "anchor_season")
POLICIES = ((None, 0), (120, 0), (250, 0), (500, 0), (250, 3), (250, 5))


def _benefit(series, index, h: int) -> np.ndarray:
    values = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, h)
        if value is not None:
            values[row] = value
    return values


def main() -> None:
    _X, _names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    with (OUT / "candidate_outputs_h5_v2.pkl").open("rb") as fh:
        outputs = pickle.load(fh)

    locked_rows = []
    audit_rows = []
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        benefit = _benefit(series, index, h)
        choices = []
        for name in ANCHORS:
            for rate in TARGET_RATES:
                for rolling, cooldown in POLICIES:
                    row = evaluate(
                        outputs[name], y, dates, currencies, benefit,
                        REGIME_VALID_YEARS, rate, rolling, cooldown,
                    )
                    row["candidate"] = name
                    choices.append(row)
        validation = pd.DataFrame(choices)
        validation.to_csv(OUT / f"anchor_postshock_selection_h{h}.csv", index=False)
        winner = best_row(validation)
        final = evaluate(
            outputs[winner.candidate], y, dates, currencies, benefit,
            FINAL_TEST_YEARS, float(winner.rate_target),
            int(winner.rolling_window) or None, int(winner.cooldown_days),
        )
        final.update({
            "h": h,
            "candidate": winner.candidate,
            "selected_validation_lift": float(winner.lift),
            "selected_validation_frequency": float(winner.frequency),
            "selection": "locked_on_2022_2023",
        })
        locked_rows.append(final)

        # A fixed family member is retained as a diagnostic across horizons.
        # It is not the locked winner because h=5 final inspection motivated it.
        diagnostic = evaluate(
            outputs["anchor_trend"], y, dates, currencies, benefit,
            FINAL_TEST_YEARS, .25, None, 0,
        )
        diagnostic.update({
            "h": h, "candidate": "anchor_trend", "selection": "posthoc_diagnostic",
            "selected_validation_lift": np.nan,
            "selected_validation_frequency": np.nan,
        })
        audit_rows.append(diagnostic)

    locked = pd.DataFrame(locked_rows)
    audit = pd.DataFrame(audit_rows)
    locked.to_csv(OUT / "locked_all_horizons.csv", index=False)
    audit.to_csv(OUT / "diagnostic_anchor_all_horizons.csv", index=False)
    columns = [
        "h", "candidate", "rate_target", "rolling_window", "cooldown_days",
        "frequency", "lift", "forward_benefit_bps", "year_lift_min",
        "corridor_lift_min", "selected_validation_lift",
    ]
    print("LOCKED")
    print(locked[columns].to_string(index=False))
    print("\nPOSTHOC DIAGNOSTIC")
    print(audit[columns[:-1]].to_string(index=False))


if __name__ == "__main__":
    main()
