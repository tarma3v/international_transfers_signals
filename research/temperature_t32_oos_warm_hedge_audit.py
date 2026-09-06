"""Independent audit for T32 disjoint warm-start Hedge outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t31_mature_fixed_share import (
    CANDIDATES,
    EXPERTS,
    PRIORITY,
    SPECS,
)
from research.temperature_t32_oos_warm_hedge import MODELS


OUT = Path("results/research/temperature/t32_oos_warm_hedge")


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

    warmup = pd.read_csv(OUT / "warmup_predictions.csv.gz")
    development = pd.read_csv(OUT / "development_predictions.csv.gz")
    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    states = pd.read_csv(OUT / "states.csv.gz")
    fit_log = pd.read_csv(OUT / "fit_log.csv")
    map_log = pd.read_csv(OUT / "map_log.csv")
    anchor_log = pd.read_csv(OUT / "warm_anchor_log.csv")
    metrics = pd.read_csv(OUT / "metrics.csv")
    nested = pd.read_csv(OUT / "screen_validation.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")

    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("open period marked as selector data")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("opened history mislabeled fresh")
    if metadata["open_period_previously_inspected"] is not True:
        raise AssertionError("previous inspection not disclosed")
    if tuple(metadata["experts"]) != EXPERTS:
        raise AssertionError("expert order mismatch")
    if set(states.candidate) != set(CANDIDATES):
        raise AssertionError("candidate state grid mismatch")
    if not (fit_log.latest_training_maturity_ord < fit_log.cutoff_ord).all():
        raise AssertionError("rank fit uses immature label")
    if not (map_log.latest_mapping_maturity_ord < map_log.cutoff_ord).all():
        raise AssertionError("mapping fit uses immature label")
    if not (map_log.slope > 0.0).all():
        raise AssertionError("mapping is not monotone")
    if not (anchor_log.latest_training_maturity_ord < anchor_log.cutoff_ord).all():
        raise AssertionError("Q4 anchor uses immature label")
    if not (pd.to_datetime(map_log.map_end_exclusive).max()
            < pd.to_datetime(warmup.query_date).min()):
        raise AssertionError("mapping and warm-up are not separated")

    all_predictions = pd.concat(
        [warmup, development, predictions], ignore_index=True)
    for frame in (warmup, development, predictions):
        values = frame[[name for name in (*MODELS, *EXPERTS) if name in frame]]
        if not np.isfinite(values.to_numpy()).all():
            raise AssertionError("non-finite saved probability")
        if not ((values >= 0.0) & (values <= 1.0)).all().all():
            raise AssertionError("probability outside [0,1]")
        if frame[["query_date", "currency"]].duplicated().any():
            raise AssertionError("publication key duplicated")
    if all_predictions[["query_date", "currency"]].duplicated().any():
        raise AssertionError("chronological output partitions overlap")

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
    if not np.allclose(predictions.t32_candidate, expected_candidate):
        raise AssertionError("screen candidate prediction mismatch")
    if not np.allclose(predictions.t32_selected, expected_selected):
        raise AssertionError("final prediction mismatch")

    weight_columns = ["weight_identity", "weight_all", "weight_recent2y"]
    if (states[weight_columns] < 0.0).any().any():
        raise AssertionError("negative online weight")
    if not np.allclose(states[weight_columns].sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("online weights do not sum to one")
    if states[["query_date", "candidate"]].duplicated().any():
        raise AssertionError("duplicate state per query/candidate")
    for _name, part in states.groupby("candidate"):
        if not part.sort_values("query_date").consumed_feedback_batches.is_monotonic_increasing:
            raise AssertionError("feedback counter decreases")
    used = states[states.latest_feedback_maturity_ord.notna()].copy()
    if not (pd.to_datetime(used.latest_feedback_publication_date)
            < pd.to_datetime(used.query_date)).all():
        raise AssertionError("same/future publication used as feedback")
    if not (used.latest_feedback_maturity_ord < used.cutoff_ord).all():
        raise AssertionError("immature feedback used")

    for name in CANDIDATES:
        candidate_states = states[states.candidate.eq(name)][[
            "query_date", *weight_columns]]
        rebuilt = all_predictions[[
            "query_date", "currency", name, *EXPERTS]].merge(
                candidate_states, on="query_date", how="left",
                validate="many_to_one")
        expected = (
            rebuilt.identity_early * rebuilt.weight_identity
            + rebuilt.all_platt_b050 * rebuilt.weight_all
            + rebuilt.recent2y_platt_b100 * rebuilt.weight_recent2y)
        if not np.allclose(rebuilt[name], expected, atol=1e-12):
            raise AssertionError(f"mixture cannot be rebuilt: {name}")

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
    first_screen = states[pd.to_datetime(states.query_date).dt.year.eq(2023)]
    if first_screen.consumed_feedback_batches.min() <= 0:
        raise AssertionError("screen still begins as a cold start")

    result = {
        "source_hashes_match": True,
        "rank_map_anchor_training_causal": True,
        "mapping_warmup_disjoint": True,
        "screen_begins_with_mature_warm_feedback": True,
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
