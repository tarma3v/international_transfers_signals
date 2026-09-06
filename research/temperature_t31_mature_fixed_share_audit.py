"""Independent audit for the T31 mature-only fixed-share experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t31_mature_fixed_share import (
    CANDIDATES,
    EXPERTS,
    MODELS,
    PRIORITY,
    SPECS,
)


OUT = Path("results/research/temperature/t31_mature_fixed_share")
T30_OUT = Path("results/research/temperature/t30_presvo_rank_postsvo_map")


def _rebuild_choice(rows):
    feasible = [row for row in rows if bool(row["feasible"])]
    return (sorted(feasible, key=lambda row: (
        row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "identity_early")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    development = pd.read_csv(OUT / "development_predictions.csv.gz")
    states = pd.read_csv(OUT / "states.csv.gz")
    metrics = pd.read_csv(OUT / "metrics.csv")
    nested = pd.read_csv(OUT / "screen_validation.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")
    t30_development = pd.read_csv(T30_OUT / "development_predictions.csv.gz")
    t30_predictions = pd.read_csv(T30_OUT / "predictions.csv.gz")

    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("open period marked as selector data")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("opened history mislabeled fresh")
    if metadata["pre2025_previously_inspected"] is not True:
        raise AssertionError("previous inspection not disclosed")
    if tuple(metadata["experts"]) != EXPERTS:
        raise AssertionError("expert order mismatch")
    if set(metadata["specs"]) != set(CANDIDATES):
        raise AssertionError("candidate grid mismatch")
    for name in CANDIDATES:
        if metadata["specs"][name] != SPECS[name]:
            raise AssertionError(f"candidate spec mismatch: {name}")

    for frame in (predictions, development):
        columns = [name for name in MODELS if name in frame]
        values = frame[columns]
        if not np.isfinite(values.to_numpy()).all():
            raise AssertionError("non-finite saved probability")
        if not ((values >= 0.0) & (values <= 1.0)).all().all():
            raise AssertionError("probability outside [0,1]")
        if frame[["query_date", "currency"]].duplicated().any():
            raise AssertionError("publication key duplicated")

    if set(nested.stage) != {"screen", "validation"}:
        raise AssertionError("nested stages incomplete")
    if set(nested.model) != set(CANDIDATES):
        raise AssertionError("nested candidate grid incomplete")
    screen_rows = nested[nested.stage.eq("screen")].to_dict("records")
    rebuilt_screen = _rebuild_choice(screen_rows)
    if rebuilt_screen != metadata["screen_selected"]:
        raise AssertionError("screen selection cannot be rebuilt")
    validation = nested[nested.stage.eq("validation")].set_index("model")
    rebuilt_validation = bool(
        rebuilt_screen != "identity_early"
        and validation.loc[rebuilt_screen, "feasible"])
    if rebuilt_validation != metadata["validation_passed"]:
        raise AssertionError("validation decision cannot be rebuilt")
    rebuilt_final = rebuilt_screen if rebuilt_validation else "identity_early"
    if rebuilt_final != metadata["selected_model"]:
        raise AssertionError("final decision cannot be rebuilt")

    expected_candidate = (
        predictions.identity_early if rebuilt_screen == "identity_early"
        else predictions[rebuilt_screen])
    expected_selected = (
        predictions.identity_early if rebuilt_final == "identity_early"
        else predictions[rebuilt_final])
    if not np.allclose(predictions.t31_candidate, expected_candidate):
        raise AssertionError("screen candidate prediction mismatch")
    if not np.allclose(predictions.t31_selected, expected_selected):
        raise AssertionError("final prediction mismatch")

    weight_columns = ["weight_identity", "weight_all", "weight_recent2y"]
    if (states[weight_columns] < 0.0).any().any():
        raise AssertionError("negative online weight")
    if not np.allclose(states[weight_columns].sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("online weights do not sum to one")
    if states[["query_date", "candidate"]].duplicated().any():
        raise AssertionError("duplicate state per query/candidate")
    if states.groupby("candidate").consumed_feedback_batches.apply(
            lambda values: values.is_monotonic_increasing).eq(False).any():
        raise AssertionError("feedback counter decreases")
    used = states[states.latest_feedback_maturity_ord.notna()].copy()
    used["query_day"] = pd.to_datetime(used.query_date).dt.date
    used["feedback_day"] = pd.to_datetime(
        used.latest_feedback_publication_date).dt.date
    if not (used.feedback_day < used.query_day).all():
        raise AssertionError("same/future publication used as feedback")
    if not (used.latest_feedback_maturity_ord < used.cutoff_ord).all():
        raise AssertionError("immature feedback used")

    # Rebuild every saved mixture from independently loaded T30 experts and
    # the state that existed before the corresponding publication.
    source = pd.concat([
        t30_development[["query_date", "currency", *EXPERTS]],
        t30_predictions[["query_date", "currency", *EXPERTS]],
    ], ignore_index=True)
    all_predictions = pd.concat([development, predictions], ignore_index=True)
    for name in CANDIDATES:
        candidate_states = states[states.candidate.eq(name)][[
            "query_date", *weight_columns]]
        rebuilt = (
            all_predictions[["query_date", "currency", name]]
            .merge(source, on=["query_date", "currency"], how="left")
            .merge(candidate_states, on="query_date", how="left")
        )
        expected = (
            rebuilt.identity_early * rebuilt.weight_identity
            + rebuilt.all_platt_b050 * rebuilt.weight_all
            + rebuilt.recent2y_platt_b100 * rebuilt.weight_recent2y)
        if not np.allclose(rebuilt[name], expected, atol=1e-12):
            raise AssertionError(f"mixture cannot be rebuilt: {name}")

    split = metadata["splits"]
    if not (split["latest_screen_maturity_ord"] < split["screen_cutoff_ord"]
            and split["latest_validation_maturity_ord"]
            < split["validation_cutoff_ord"]):
        raise AssertionError("nested split contains immature labels")
    expected_intervals = {
        (comparison, metric, block)
        for comparison in ("identity", "t25")
        for metric in ("auc", "brier", "logloss")
        for block in (20, 50)
    }
    if set(zip(intervals.comparison, intervals.metric, intervals.block_dates)) != expected_intervals:
        raise AssertionError("paired interval grid incomplete")
    if len(metrics[metrics.slice.eq("ALL")]) != len(MODELS):
        raise AssertionError("overall metrics incomplete")

    result = {
        "source_hashes_match": True,
        "candidate_grid_complete": True,
        "screen_selected": rebuilt_screen,
        "validation_passed": rebuilt_validation,
        "selected_model": rebuilt_final,
        "passed": bool(metadata["passed"]),
        "feedback_strictly_mature_and_prior": True,
        "weights_finite_nonnegative_and_normalized": True,
        "all_candidate_mixtures_rebuilt": True,
        "selected_predictions_rebuilt": True,
        "publication_keys_unique": True,
        "probabilities_finite_and_bounded": True,
        "paired_interval_grid_complete": True,
        "selection_on_open_2025_2026": False,
        "fresh_independent_holdout": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
