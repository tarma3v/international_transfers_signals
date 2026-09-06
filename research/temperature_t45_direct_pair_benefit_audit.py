"""Independent reconstruction and causality audit for T45."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS
from research.temperature_t45_direct_pair_benefit import (
    BLOCKS,
    EMBARGO_DAYS,
    OUT,
    fit_quarterly_residual,
    run_experiment,
)


def _frame_same(saved: pd.DataFrame, rebuilt: pd.DataFrame):
    assert list(saved.columns) == list(rebuilt.columns)
    assert len(saved) == len(rebuilt)
    for column in saved.columns:
        if pd.api.types.is_numeric_dtype(saved[column]):
            np.testing.assert_allclose(
                saved[column].to_numpy(float), rebuilt[column].to_numpy(float),
                rtol=1e-10, atol=1e-10, equal_nan=True,
            )
        else:
            left = saved[column].fillna("").astype(str).to_numpy()
            right = rebuilt[column].fillna("").astype(str).to_numpy()
            np.testing.assert_array_equal(left, right)


def _log_same(saved: pd.DataFrame, rebuilt: pd.DataFrame):
    rebuilt = rebuilt.copy()
    rebuilt["coefficients"] = rebuilt.coefficients.map(str)
    for value in saved.coefficients.dropna():
        parsed = ast.literal_eval(value)
        assert isinstance(parsed, list)
    _frame_same(saved, rebuilt)


def audit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    rebuilt = run_experiment()
    _frame_same(pd.read_csv(OUT / "metrics.csv"), rebuilt["metrics"])
    _frame_same(
        pd.read_csv(OUT / "paired_bootstrap.csv"), rebuilt["bootstrap"])
    _frame_same(pd.read_csv(OUT / "historical_gates.csv"), rebuilt["gates"])
    _log_same(pd.read_csv(OUT / "training_log.csv"), rebuilt["logs"])
    _frame_same(
        pd.read_csv(OUT / "publication_predictions.csv.gz"),
        rebuilt["predictions"],
    )

    expected_open = [
        h for h in HORIZONS
        if rebuilt["gates"].loc[rebuilt["gates"].h.eq(h), "stage_pass"].all()
    ]
    assert expected_open == metadata["open_horizons"] == [3, 5, 10]
    open_metrics = rebuilt["metrics"].period.eq("open_2025_2026")
    assert sorted(rebuilt["metrics"].loc[open_metrics, "h"].unique()) == expected_open

    hard = rebuilt["hard"]
    for h in HORIZONS:
        raw = rebuilt["raw"][h]
        fallback = ~hard & np.isfinite(raw["baseline"])
        np.testing.assert_array_equal(
            raw["candidate"][fallback], raw["baseline"][fallback])

    count = rebuilt["panel"]["count"].fillna(0.0).to_numpy(float)
    age = rebuilt["panel"]["age_minutes"].fillna(720.0).to_numpy(float)
    expected_hard = (count >= 6.0) & (age <= 60.0)
    expected_quality = np.where(
        count > 0.0, np.minimum(count / 24.0, 1.0) * np.exp(-age / 120.0), 0.0
    )
    np.testing.assert_array_equal(hard, expected_hard)
    np.testing.assert_allclose(rebuilt["quality"], expected_quality)

    for row in rebuilt["logs"].itertuples(index=False):
        if row.model_fit:
            origin = dt.date.fromisoformat(row.origin)
            last = dt.date.fromisoformat(row.last_target_maturity)
            assert last < origin - dt.timedelta(days=EMBARGO_DAYS)

    # Physically alter every direct feature, target and maturity at/after a
    # future boundary. The exact earlier prediction prefix must stay unchanged,
    # while the later positive control must react.
    h = 5
    raw = rebuilt["raw"][h]
    boundary = dt.date(2025, 1, 1)
    future = rebuilt["dates"] >= boundary
    changed_matrix = raw["matrix"].copy()
    changed_matrix[future] = changed_matrix[future] * 17.0 + 31.0
    changed_target = rebuilt["targets"][h].copy()
    changed_target[future & np.isfinite(changed_target)] = (
        -13.0 * changed_target[future & np.isfinite(changed_target)] + 1000.0)
    changed_maturity = raw["maturity"].copy()
    changed_maturity[future] = dt.date(1900, 1, 1)
    altered = fit_quarterly_residual(
        changed_matrix, changed_target, raw["baseline"], changed_maturity,
        rebuilt["dates"], hard, rebuilt["quality"],
    )[0]
    np.testing.assert_array_equal(
        raw["candidate"][~future], altered[~future])
    affected = future & hard & np.isfinite(raw["candidate"]) & np.isfinite(altered)
    assert affected.any()
    assert np.any(raw["candidate"][affected] != altered[affected])

    assert set(rebuilt["bootstrap"].block_dates.unique()) == set(BLOCKS)
    assert metadata["production_promoted"] is False
    assert metadata["push_changed"] is False
    assert metadata["probability_changed"] is False
    assert metadata["runtime_changed"] is False
    checks = {
        "source_hashes_match": True,
        "quarterly_predictions_rebuilt": True,
        "historical_metrics_and_gates_rebuilt": True,
        "paired_bootstrap_rebuilt": True,
        "hard_gate_and_quality_formula_rebuilt": True,
        "ineligible_rows_equal_t6_exactly": True,
        "maturity_and_embargo_verified": True,
        "future_prefix_corruption_passed": True,
        "open_horizons": expected_open,
        "production_promoted": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    return checks


def main():
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
