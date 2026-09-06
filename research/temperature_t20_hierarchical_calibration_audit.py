"""Independent consistency audit for saved T20 outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("results/research/temperature/t20_hierarchical_calibration")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    states = pd.read_csv(OUT / "state_gates.csv")
    checks = pd.read_csv(OUT / "fit_checks.csv")
    models = [
        "identity", "global_platt", "fixed_logit_shrink_80",
        "hierarchical_beta",
    ]
    if len(predictions) != metadata["evaluation_rows"]:
        raise AssertionError("prediction row count mismatch")
    if len(states) != metadata["state_rows"]:
        raise AssertionError("state row count mismatch")
    if int(states["pass"].sum()) != metadata["passing_states"]:
        raise AssertionError("pass count mismatch")
    if not (checks.latest_train_maturity_ord < checks.fit_cutoff_ord).all():
        raise AssertionError("immature label in fit checks")
    if not (
        pd.to_datetime(predictions.source_at, utc=True)
        <= pd.to_datetime(predictions.query_at, utc=True)
    ).all():
        raise AssertionError("future source timestamp")
    if not np.allclose(predictions.identity, predictions.probability):
        raise AssertionError("identity control changed")
    logit = np.log(np.clip(predictions.identity, 1e-5, 1 - 1e-5)
                   / (1 - np.clip(predictions.identity, 1e-5, 1 - 1e-5)))
    expected_shrink = 1 / (1 + np.exp(-0.8 * logit))
    if not np.allclose(predictions.fixed_logit_shrink_80, expected_shrink):
        raise AssertionError("fixed shrink control mismatch")
    if predictions[models].isna().any().any():
        raise AssertionError("missing calibrated probability")
    if not ((predictions[models] >= 0.0) & (predictions[models] <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")
    result = {
        "source_hashes_match": True,
        "prediction_rows": int(len(predictions)),
        "state_rows": int(len(states)),
        "passing_states": int(states["pass"].sum()),
        "all_training_labels_mature": True,
        "all_sources_no_later_than_query": True,
        "identity_and_fixed_controls_rebuilt": True,
        "probabilities_finite_and_bounded": True,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
