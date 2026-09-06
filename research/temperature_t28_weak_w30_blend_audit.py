"""Independent consistency audit for T28 weak-w30 blend outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t28_weak_w30_blend import BLENDS, MODELS, PRIORITY


OUT = Path("results/research/temperature/t28_weak_w30_blend")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    states = pd.read_csv(OUT / "states.csv.gz")
    fit_log = pd.read_csv(OUT / "historical_fit_log.csv")
    metrics = pd.read_csv(OUT / "metrics.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")
    nested = pd.read_csv(OUT / "screen_validation.csv")

    if len(predictions) != metadata["evaluation_rows"]:
        raise AssertionError("prediction row count mismatch")
    if predictions.query_date.nunique() != metadata["evaluation_dates"]:
        raise AssertionError("evaluation date count mismatch")
    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("open evaluation marked as selector data")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("retrospective split mislabeled fresh")
    if metadata["pre2025_aggregate_previously_inspected"] is not True:
        raise AssertionError("prior pre-2025 inspection not disclosed")

    values = predictions[list(MODELS)]
    if not np.isfinite(values.to_numpy()).all():
        raise AssertionError("non-finite probability")
    if not ((values >= 0.0) & (values <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")
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

    if set(nested.stage) != {"screen", "validation"}:
        raise AssertionError("nested stages incomplete")
    if set(nested.model) != set(BLENDS):
        raise AssertionError("blend candidate grid incomplete")
    screen = nested[nested.stage.eq("screen")].to_dict("records")
    feasible = [row for row in screen if bool(row["feasible"])]
    rebuilt_screen = (
        sorted(feasible, key=lambda row: (row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "t25_base"
    )
    if rebuilt_screen != metadata["screen_selected"]:
        raise AssertionError("screen selection cannot be rebuilt")
    validation = nested[nested.stage.eq("validation")].set_index("model")
    rebuilt_validation = bool(
        rebuilt_screen != "t25_base" and validation.loc[rebuilt_screen, "feasible"])
    if rebuilt_validation != metadata["validation_passed"]:
        raise AssertionError("validation decision cannot be rebuilt")
    rebuilt_final = rebuilt_screen if rebuilt_validation else "t25_base"
    if rebuilt_final != metadata["selected_model"]:
        raise AssertionError("final selection cannot be rebuilt")
    expected = predictions.t25_base if rebuilt_final == "t25_base" else predictions[rebuilt_final]
    if not np.allclose(predictions.weak_selected, expected):
        raise AssertionError("selected prediction cannot be rebuilt")

    split = metadata["split"]
    if not (
        split["latest_screen_maturity_ord"] < split["nested_screen_cutoff_ord"]
        and split["latest_validation_maturity_ord"] < split["nested_validation_cutoff_ord"]
    ):
        raise AssertionError("nested split contains immature labels")
    if not metadata["checks"]["screen_and_validation_disjoint"]:
        raise AssertionError("screen and validation overlap")
    if not (fit_log.latest_training_maturity_ord < fit_log.cutoff_ord).all():
        raise AssertionError("historical compact fit uses immature labels")
    used = states[states.latest_feedback_maturity_ord.notna()]
    if not (used.latest_feedback_maturity_ord < used.cutoff_ord).all():
        raise AssertionError("update state uses future feedback")

    expected_grid = {
        (comparison, metric, block)
        for comparison in ("t25", "identity")
        for metric in ("auc", "brier", "logloss")
        for block in (20, 50)
    }
    if set(zip(intervals.comparison, intervals.metric, intervals.block_dates)) != expected_grid:
        raise AssertionError("paired interval grid incomplete")
    if len(metrics[metrics.slice.eq("ALL")]) != len(MODELS):
        raise AssertionError("overall metrics incomplete")

    result = {
        "source_hashes_match": True,
        "prediction_rows": int(len(predictions)),
        "evaluation_dates": int(predictions.query_date.nunique()),
        "screen_selected": rebuilt_screen,
        "validation_passed": rebuilt_validation,
        "selected_model": rebuilt_final,
        "passed": bool(metadata["passed"]),
        "nested_masks_disjoint_and_mature": True,
        "historical_fit_causal": True,
        "feedback_trace_causal": True,
        "selected_prediction_rebuilt": True,
        "probabilities_finite_and_bounded": True,
        "paired_interval_grid_complete": True,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
