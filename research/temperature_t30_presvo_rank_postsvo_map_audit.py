"""Independent audit for T30 pre-SVO rank/post-SVO map outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from research.temperature_t21_h20_curve_head import _logit
from research.temperature_t30_presvo_rank_postsvo_map import (
    BETAS,
    CANDIDATES,
    FAMILIES,
    MODELS,
    PRIORITY,
)


OUT = Path("results/research/temperature/t30_presvo_rank_postsvo_map")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    development = pd.read_csv(OUT / "development_predictions.csv.gz")
    fit_log = pd.read_csv(OUT / "fit_log.csv")
    map_log = pd.read_csv(OUT / "map_log.csv")
    metrics = pd.read_csv(OUT / "metrics.csv")
    nested = pd.read_csv(OUT / "screen_validation.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")

    if metadata["selection_on_open_2025_2026"] is not False:
        raise AssertionError("open period marked as selector data")
    if metadata["fresh_independent_holdout"] is not False:
        raise AssertionError("opened history mislabeled fresh")
    if metadata["open_period_previously_inspected"] is not True:
        raise AssertionError("prior inspection not disclosed")
    if set(metadata["candidate_grid"]) != set(CANDIDATES):
        raise AssertionError("candidate grid mismatch")
    if set(fit_log.family) != set(FAMILIES):
        raise AssertionError("rank family log incomplete")
    if set(map_log.family) != set(FAMILIES):
        raise AssertionError("map family log incomplete")
    if not (fit_log.latest_training_maturity_ord < fit_log.cutoff_ord).all():
        raise AssertionError("rank fit uses immature labels")
    if not (map_log.latest_mapping_maturity_ord < map_log.cutoff_ord).all():
        raise AssertionError("mapping fit uses immature labels")
    if not (map_log.slope > 0).all():
        raise AssertionError("non-positive Platt slope")

    for frame in (predictions, development):
        columns = [name for name in MODELS if name in frame]
        values = frame[columns]
        if not np.isfinite(values.to_numpy()).all():
            raise AssertionError("non-finite saved probability")
        if not ((values >= 0.0) & (values <= 1.0)).all().all():
            raise AssertionError("probability outside [0,1]")
        if frame[["query_date", "currency"]].duplicated().any():
            raise AssertionError("publication key duplicated")

    if predictions.query_date.nunique() != metadata["splits"]["evaluation_dates"]:
        raise AssertionError("evaluation date count mismatch")
    if len(predictions) != metadata["splits"]["evaluation_rows"]:
        raise AssertionError("evaluation row count mismatch")

    if set(nested.stage) != {"screen", "validation"}:
        raise AssertionError("nested stages incomplete")
    if set(nested.model) != set(CANDIDATES):
        raise AssertionError("nested candidate grid incomplete")
    screen = nested[nested.stage.eq("screen")].to_dict("records")
    feasible = [row for row in screen if bool(row["feasible"])]
    rebuilt_screen = (sorted(feasible, key=lambda row: (
        row["brier"], PRIORITY[row["model"]]))[0]["model"]
        if feasible else "identity_early")
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
        raise AssertionError("final model cannot be rebuilt")

    expected_candidate = (
        predictions.identity_early if rebuilt_screen == "identity_early"
        else predictions[rebuilt_screen])
    expected_selected = (
        predictions.identity_early if rebuilt_final == "identity_early"
        else predictions[rebuilt_final])
    if not np.allclose(predictions.t30_candidate, expected_candidate):
        raise AssertionError("screen candidate prediction mismatch")
    if not np.allclose(predictions.t30_selected, expected_selected):
        raise AssertionError("final prediction mismatch")

    # Every map is a positive affine transform of its raw logit; verify the
    # saved values and monotonicity independently from the fitting code.
    map_index = map_log.set_index("family")
    for family in FAMILIES:
        raw_name = f"{family}_raw"
        raw = predictions[raw_name].to_numpy()
        order = np.argsort(raw, kind="mergesort")
        for beta in BETAS:
            name = f"{family}_platt_b{int(beta * 100):03d}"
            row = map_index.loc[family]
            platt = expit(row.intercept + row.slope * _logit(raw))
            rebuilt = expit((1.0 - beta) * _logit(raw) + beta * _logit(platt))
            if not np.allclose(predictions[name], rebuilt, atol=1e-12):
                raise AssertionError(f"map formula mismatch: {name}")
            if np.diff(predictions[name].to_numpy()[order]).min() < -1e-12:
                raise AssertionError(f"map is not monotone: {name}")

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
    if metadata["checks"]["t4_query_anchor_max_abs_error"] > 1e-12:
        raise AssertionError("saved T4/query anchor mismatch")

    result = {
        "source_hashes_match": True,
        "screen_selected": rebuilt_screen,
        "validation_passed": rebuilt_validation,
        "selected_model": rebuilt_final,
        "passed": bool(metadata["passed"]),
        "rank_and_mapping_fits_causal": True,
        "all_candidate_maps_monotone": True,
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
