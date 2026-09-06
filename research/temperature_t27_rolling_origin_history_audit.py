"""Independent consistency audit for T27 rolling-origin history outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t26_delayed_base_rate import CANDIDATES, MODELS, PRIORITY


OUT = Path("results/research/temperature/t27_rolling_origin_history")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    history = pd.read_csv(OUT / "historical_feedback.csv.gz")
    fit_log = pd.read_csv(OUT / "historical_fit_log.csv")
    states = pd.read_csv(OUT / "states.csv.gz")
    metrics = pd.read_csv(OUT / "metrics.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")

    if len(predictions) != metadata["evaluation_rows"]:
        raise AssertionError("prediction row count mismatch")
    if predictions.query_date.nunique() != metadata["evaluation_dates"]:
        raise AssertionError("evaluation date count mismatch")
    if metadata["selection_on_open_period"] is not False:
        raise AssertionError("open evaluation marked as selection data")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("opened evaluation mislabeled fresh")
    if metadata["open_period_previously_inspected"] is not True:
        raise AssertionError("prior inspection not disclosed")
    if metadata["t25_base_model"] != "residual_a040":
        raise AssertionError("T25 base selection changed")

    if len(history) != 1235 or history.query_date.nunique() != 247:
        raise AssertionError("unexpected 2023 history coverage")
    if set(pd.to_datetime(history.query_date).dt.year) != {2023}:
        raise AssertionError("historical feedback escaped 2023")
    if history[["publication_date", "currency"]].duplicated().any():
        raise AssertionError("historical feedback key duplicated")
    if history.t4_model_n_train.min() < 500:
        raise AssertionError("T4 OOS history lacks training support")
    if not np.isfinite(history[["identity_early", "t25_base"]]).all().all():
        raise AssertionError("non-finite historical probability")
    if not ((history[["identity_early", "t25_base"]] >= 0.0)
            & (history[["identity_early", "t25_base"]] <= 1.0)).all().all():
        raise AssertionError("historical probability outside [0,1]")

    if len(fit_log) != 4:
        raise AssertionError("quarterly compact fit grid incomplete")
    if not (
        fit_log.latest_training_maturity_ord < fit_log.cutoff_ord
    ).all():
        raise AssertionError("immature label in compact fit")
    if not (
        pd.to_datetime(fit_log.latest_training_date)
        < pd.to_datetime(fit_log.origin)
    ).all():
        raise AssertionError("future row in compact fit")

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
        raise AssertionError("non-finite evaluation probability")
    if not ((values >= 0.0) & (values <= 1.0)).all().all():
        raise AssertionError("evaluation probability outside [0,1]")

    selected = metadata["selected_model"]
    expected = predictions.t25_base if selected == "t25_base" else predictions[selected]
    if not np.allclose(predictions.delayed_selected, expected):
        raise AssertionError("selected column cannot be rebuilt")
    feasible = [row for row in metadata["screen"] if row["feasible"]]
    rebuilt = (
        sorted(feasible, key=lambda row: (row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "t25_base"
    )
    if selected != rebuilt:
        raise AssertionError("selector cannot be rebuilt")
    if set(row["model"] for row in metadata["screen"]) != set(CANDIDATES):
        raise AssertionError("candidate screen incomplete")

    if states[["query_date", "candidate", "currency"]].duplicated().any():
        raise AssertionError("duplicate update state")
    used = states[states.latest_feedback_maturity_ord.notna()]
    if not (used.latest_feedback_maturity_ord < used.cutoff_ord).all():
        raise AssertionError("future feedback in update state")
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
        "historical_rows": int(len(history)),
        "historical_dates": int(history.query_date.nunique()),
        "historical_quarterly_fits_causal": True,
        "historical_feedback_unique": True,
        "feedback_trace_causal": True,
        "selected_model": selected,
        "passed": bool(metadata["passed"]),
        "selection_rebuilt": True,
        "selected_prediction_rebuilt": True,
        "probabilities_finite_and_bounded": True,
        "paired_interval_grid_complete": True,
        "selection_on_open_period": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
