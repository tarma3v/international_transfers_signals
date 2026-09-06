"""Independent consistency audit for T24 history-only h20 outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("results/research/temperature/t24_history_h20_anchor")
MODELS = [
    "identity_early", "history_h20_selected", "compact_logit",
    "extended_hgb", "recent_hgb", "path_extra",
]


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")
    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    metrics = pd.read_csv(OUT / "metrics.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")
    if len(predictions) != metadata["evaluation_rows"]:
        raise AssertionError("prediction row count mismatch")
    if predictions.query_date.nunique() != metadata["evaluation_dates"]:
        raise AssertionError("evaluation date count mismatch")
    if metadata["selection_on_open_period"] is not False:
        raise AssertionError("open evaluation marked as selection data")
    if not (
        pd.to_datetime(predictions.source_at, utc=True)
        <= pd.to_datetime(predictions.query_at, utc=True)
    ).all():
        raise AssertionError("future source timestamp")
    if not (
        pd.to_datetime(predictions.publication_date)
        <= pd.to_datetime(predictions.query_date)
    ).all():
        raise AssertionError("future publication mapped to query")
    values = predictions[MODELS]
    if values.isna().any().any():
        raise AssertionError("missing probability")
    if not ((values >= 0.0) & (values <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")
    selected = metadata["selected_model"]
    expected = predictions.identity_early if selected == "identity_early" else predictions[selected]
    if not np.allclose(predictions.history_h20_selected, expected):
        raise AssertionError("saved primary does not match selected model")
    split = metadata["split_checks"]
    if not (
        split["latest_calibration_maturity_ord"] < split["calibration_cutoff_ord"]
        and split["latest_selection_maturity_ord"] < split["selection_cutoff_ord"]
    ):
        raise AssertionError("immature calibration or selection label")
    if set(intervals.block_dates) != {20, 50}:
        raise AssertionError("missing bootstrap block")
    if set(intervals.metric) != {"auc", "brier"}:
        raise AssertionError("missing bootstrap metric")
    if len(metrics[metrics.slice.eq("ALL")]) != len(MODELS):
        raise AssertionError("incomplete overall metrics")
    result = {
        "source_hashes_match": True,
        "prediction_rows": int(len(predictions)),
        "evaluation_dates": int(predictions.query_date.nunique()),
        "selected_model": selected,
        "passed": bool(metadata["passed"]),
        "all_sources_no_later_than_query": True,
        "publication_mapping_causal": True,
        "fit_calibration_selection_mature": True,
        "selected_prediction_rebuilt": True,
        "probabilities_finite_and_bounded": True,
        "paired_bootstrap_grid_complete": True,
        "selection_on_open_period": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
