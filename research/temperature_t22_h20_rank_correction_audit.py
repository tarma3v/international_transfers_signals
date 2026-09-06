"""Independent consistency audit for saved T22 outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t22_h20_rank_correction import ALPHAS, PRIMARY, _alpha_name


OUT = Path("results/research/temperature/t22_h20_rank_correction")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    states = pd.read_csv(OUT / "state_gates.csv")
    fits = pd.read_csv(OUT / "fit_checks.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")
    details = json.loads((OUT / "model_details.json").read_text())

    if len(predictions) != metadata["evaluation_rows"]:
        raise AssertionError("prediction row count mismatch")
    if len(states) != metadata["state_rows"]:
        raise AssertionError("state row count mismatch")
    if int(states["pass"].sum()) != metadata["passing_states"]:
        raise AssertionError("pass count mismatch")
    if metadata["selection_on_open_period"] is not False:
        raise AssertionError("open evaluation was marked as selection data")
    if not (fits.latest_base_maturity_ord < fits.base_cutoff_ord).all():
        raise AssertionError("immature base label")
    if not (
        fits.latest_calibration_maturity_ord < fits.eval_cutoff_ord
    ).all():
        raise AssertionError("immature calibration label")
    if not (
        pd.to_datetime(predictions.source_at, utc=True)
        <= pd.to_datetime(predictions.query_at, utc=True)
    ).all():
        raise AssertionError("future source timestamp")
    if not np.allclose(predictions.identity_h20, predictions.p20):
        raise AssertionError("identity differs from frozen h20")

    fixed = [_alpha_name(alpha) for alpha in ALPHAS]
    values = predictions[["identity_h20", PRIMARY, *fixed]]
    if values.isna().any().any():
        raise AssertionError("missing probability")
    if not ((values >= 0.0) & (values <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")
    if not np.allclose(predictions[_alpha_name(0.0)], predictions.identity_h20):
        raise AssertionError("zero correction does not equal identity")

    fit_lookup = fits.set_index(["scenario", "clock"])
    for (scenario, clock), part in predictions.groupby(["scenario", "clock"]):
        alpha = float(fit_lookup.loc[(scenario, clock), "selected_alpha"])
        if not np.allclose(part[PRIMARY], part[_alpha_name(alpha)]):
            raise AssertionError("primary does not match selected fixed alpha")

    for detail in details:
        feasible = [row for row in detail["calibration_grid"] if row["feasible"]]
        expected = sorted(
            feasible, key=lambda row: (-row["auc"], row["alpha"])
        )[0]["alpha"] if feasible else 0.0
        if not np.isclose(detail["selected_alpha"], expected):
            raise AssertionError("alpha selection was not rebuilt")

    interval_keys = intervals[
        ["scenario", "clock", "metric", "block_dates"]
    ].drop_duplicates()
    if len(interval_keys) != 40 * 2 * 2:
        raise AssertionError("incomplete paired-bootstrap grid")
    result = {
        "source_hashes_match": True,
        "prediction_rows": int(len(predictions)),
        "state_rows": int(len(states)),
        "passing_states": int(states["pass"].sum()),
        "all_fit_labels_mature": True,
        "all_sources_no_later_than_query": True,
        "zero_alpha_rebuilds_identity": True,
        "selected_probabilities_rebuilt": True,
        "calibration_only_alpha_selection_rebuilt": True,
        "probabilities_finite_and_bounded": True,
        "paired_bootstrap_grid_complete": True,
        "selection_on_open_period": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
