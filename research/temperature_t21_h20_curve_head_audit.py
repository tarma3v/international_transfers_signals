"""Independent consistency audit for saved T21 outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("results/research/temperature/t21_h20_curve_head")
MODELS = [
    "identity_h20", "curve_logit", "curve_hgb_platt", "curve_extra_platt",
]


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
        raise AssertionError("identity control differs from frozen h20")
    values = predictions[MODELS]
    if values.isna().any().any():
        raise AssertionError("missing probability")
    if not ((values >= 0.0) & (values <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")
    expected_states = (
        predictions[["scenario", "clock"]].drop_duplicates().shape[0]
    )
    if expected_states != 40:
        raise AssertionError("unexpected scenario-clock state count")
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
        "identity_control_rebuilt": True,
        "probabilities_finite_and_bounded": True,
        "paired_bootstrap_grid_complete": True,
        "selection_on_open_period": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
