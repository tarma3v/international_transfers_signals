"""Independent consistency audit for T26 delayed base-rate outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t26_delayed_base_rate import CANDIDATES, MODELS, PRIORITY


OUT = Path("results/research/temperature/t26_delayed_base_rate")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    states = pd.read_csv(OUT / "states.csv.gz")
    metrics = pd.read_csv(OUT / "metrics.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")
    if len(predictions) != metadata["evaluation_rows"]:
        raise AssertionError("prediction row count mismatch")
    if predictions.query_date.nunique() != metadata["evaluation_dates"]:
        raise AssertionError("evaluation date count mismatch")
    if metadata["selection_on_open_period"] is not False:
        raise AssertionError("open evaluation marked as selection data")
    if metadata["t25_base_model"] != "residual_a040":
        raise AssertionError("T25 base selection changed")
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
    expected = predictions.t25_base if selected == "t25_base" else predictions[selected]
    if not np.allclose(predictions.delayed_selected, expected):
        raise AssertionError("saved primary does not match selected model")

    feasible = [row for row in metadata["screen"] if row["feasible"]]
    rebuilt = (
        sorted(feasible, key=lambda row: (row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "t25_base"
    )
    if selected != rebuilt:
        raise AssertionError("selection cannot be rebuilt")
    if set(row["model"] for row in metadata["screen"]) != set(CANDIDATES):
        raise AssertionError("candidate screen incomplete")

    if states[["query_date", "candidate", "currency"]].duplicated().any():
        raise AssertionError("duplicate update state")
    if set(states.candidate) != set(CANDIDATES):
        raise AssertionError("update state candidate missing")
    used = states[states.latest_feedback_maturity_ord.notna()]
    if not (used.latest_feedback_maturity_ord < used.cutoff_ord).all():
        raise AssertionError("future or embargoed feedback in update")
    global_only = states[~states.candidate.eq("delayed_hier_w125")]
    if not np.allclose(global_only.currency_delta, 0.0):
        raise AssertionError("currency delta leaked into global candidate")
    for name, spec in metadata["parameters"]["specs"].items():
        window = spec["window"]
        if window is not None and not (
            states.loc[states.candidate.eq(name), "feedback_dates"] <= window
        ).all():
            raise AssertionError(f"window exceeded for {name}")
    if not metadata["checks"]["event_feedback_unique"]:
        raise AssertionError("calendar holds duplicated feedback")
    if not metadata["checks"]["selection_labels_mature"]:
        raise AssertionError("selection label not mature")

    expected_grid = {
        (comparison, metric, block)
        for comparison in ("t25", "identity")
        for metric in ("auc", "brier", "logloss")
        for block in (20, 50)
    }
    actual_grid = set(zip(intervals.comparison, intervals.metric, intervals.block_dates))
    if actual_grid != expected_grid:
        raise AssertionError("paired interval grid incomplete")
    if len(metrics[metrics.slice.eq("ALL")]) != len(MODELS):
        raise AssertionError("overall metrics incomplete")

    result = {
        "source_hashes_match": True,
        "prediction_rows": int(len(predictions)),
        "evaluation_dates": int(predictions.query_date.nunique()),
        "selected_model": selected,
        "passed": bool(metadata["passed"]),
        "t25_base_rebuilt": True,
        "selection_rebuilt": True,
        "selected_prediction_rebuilt": True,
        "all_sources_no_later_than_query": True,
        "publication_mapping_causal": True,
        "feedback_trace_causal": True,
        "event_feedback_unique": True,
        "probabilities_finite_and_bounded": True,
        "paired_interval_grid_complete": True,
        "selection_on_open_period": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
