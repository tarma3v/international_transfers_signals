"""Independent consistency audit for T25 anchor-preserving maps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t25_anchor_preserving_map import (
    CANDIDATES,
    MODELS,
    PRIORITY,
)


OUT = Path("results/research/temperature/t25_anchor_preserving_map")


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

    values = predictions[list(MODELS)]
    if not np.isfinite(values.to_numpy()).all():
        raise AssertionError("non-finite probability")
    if not ((values >= 0.0) & (values <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")

    selected = metadata["selected_model"]
    expected = (
        predictions.identity_early
        if selected == "identity_early" else predictions[selected]
    )
    if not np.allclose(predictions.anchor_map_selected, expected):
        raise AssertionError("saved primary does not match selected map")
    feasible = [row for row in metadata["screen"] if row["feasible"]]
    rebuilt_selected = (
        sorted(
            feasible,
            key=lambda row: (-row["auc"], PRIORITY[row["model"]]),
        )[0]["model"]
        if feasible else "identity_early"
    )
    if selected != rebuilt_selected:
        raise AssertionError("selection cannot be rebuilt from frozen screen")
    if not set(row["model"] for row in metadata["screen"]) == set(CANDIDATES):
        raise AssertionError("candidate screen incomplete")

    max_multiset_error = 0.0
    for _, part in predictions.groupby("query_date", sort=False):
        error = float(np.max(np.abs(
            np.sort(part.daily_permute.to_numpy())
            - np.sort(part.identity_early.to_numpy())
        )))
        max_multiset_error = max(max_multiset_error, error)
    if max_multiset_error >= 1e-12:
        raise AssertionError("daily permutation changed the anchor multiset")
    if not metadata["checks"]["mapping_reference_labels_mature"]:
        raise AssertionError("immature mapping-reference label")
    if not metadata["checks"]["selection_labels_mature"]:
        raise AssertionError("immature selection label")
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
        "mapping_reference_and_selection_mature": True,
        "selection_rebuilt": True,
        "selected_prediction_rebuilt": True,
        "probabilities_finite_and_bounded": True,
        "daily_permute_max_multiset_error": max_multiset_error,
        "paired_bootstrap_grid_complete": True,
        "selection_on_open_period": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
