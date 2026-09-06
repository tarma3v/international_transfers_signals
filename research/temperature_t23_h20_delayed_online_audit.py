"""Independent consistency audit for saved T23 delayed calibration outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.temperature_t23_h20_delayed_online import PRIORITY, PRIMARY


OUT = Path("results/research/temperature/t23_h20_delayed_online")


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}")

    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    states = pd.read_csv(OUT / "state_gates.csv")
    logs = pd.read_csv(OUT / "monthly_logs.csv")
    selection = pd.read_csv(OUT / "selection_metrics.csv")
    intervals = pd.read_csv(OUT / "paired_bootstrap.csv")
    if len(predictions) != metadata["evaluation_rows"]:
        raise AssertionError("prediction row count mismatch")
    if len(states) != metadata["state_rows"]:
        raise AssertionError("state row count mismatch")
    if int(states["pass"].sum()) != metadata["passing_states"]:
        raise AssertionError("pass count mismatch")
    if metadata["selection_on_open_period"] is not False:
        raise AssertionError("open evaluation marked as selection data")
    if not (
        logs.latest_eligible_maturity_ord < logs.fit_cutoff_ord
    ).all():
        raise AssertionError("monthly mapper used immature label")
    if not (
        pd.to_datetime(predictions.source_at, utc=True)
        <= pd.to_datetime(predictions.query_at, utc=True)
    ).all():
        raise AssertionError("future source timestamp")
    if not np.allclose(predictions.identity_h20, predictions.p20):
        raise AssertionError("identity differs from frozen h20")
    values = predictions[
        ["identity_h20", PRIMARY, "delayed_anchor", "delayed_joint"]]
    if values.isna().any().any():
        raise AssertionError("missing probability")
    if not ((values >= 0.0) & (values <= 1.0)).all().all():
        raise AssertionError("probability outside [0,1]")
    query_month = pd.to_datetime(predictions.query_date).dt.to_period("M")
    origin_month = pd.to_datetime(predictions.monthly_origin).dt.to_period("M")
    if not (query_month == origin_month).all():
        raise AssertionError("prediction mapped with another month origin")
    expected = np.where(
        predictions.selected_family.eq("identity"), predictions.identity_h20,
        np.where(
            predictions.selected_family.eq("anchor_logit"),
            predictions.delayed_anchor, predictions.delayed_joint))
    if not np.allclose(predictions[PRIMARY], expected):
        raise AssertionError("selected prediction does not match family")

    for key, part in selection.groupby(
            ["scenario", "clock", "monthly_origin"], sort=False):
        feasible = part[part.feasible.astype(bool)].copy()
        rebuilt = sorted(
            feasible.to_dict("records"),
            key=lambda row: (-row["auc"], PRIORITY[row["family"]])
        )[0]["family"]
        logged = logs.set_index(
            ["scenario", "clock", "monthly_origin"]).loc[key, "selected_family"]
        if rebuilt != logged:
            raise AssertionError("monthly family selection mismatch")

    interval_keys = intervals[
        ["scenario", "clock", "metric", "block_dates"]
    ].drop_duplicates()
    if len(interval_keys) != 40 * 2 * 2:
        raise AssertionError("incomplete paired-bootstrap grid")
    result = {
        "source_hashes_match": True,
        "prediction_rows": int(len(predictions)),
        "monthly_updates": int(len(logs)),
        "state_rows": int(len(states)),
        "passing_states": int(states["pass"].sum()),
        "all_monthly_labels_mature": True,
        "all_sources_no_later_than_query": True,
        "monthly_origins_match_queries": True,
        "selected_predictions_rebuilt": True,
        "monthly_selection_rebuilt": True,
        "probabilities_finite_and_bounded": True,
        "paired_bootstrap_grid_complete": True,
        "selection_on_open_period": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
