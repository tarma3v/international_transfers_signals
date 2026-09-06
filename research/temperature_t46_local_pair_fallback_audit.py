"""Independent reconstruction and prefix audit for T46."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from ml.targets import HORIZONS
from research.temperature_t46_local_pair_fallback import (
    EMBARGO_DAYS,
    OUT,
    fit_local_heads,
    loadz,
    run_experiment,
)


def same(left, right):
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12,
                               equal_nan=True)


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest, path

    rebuilt = run_experiment()
    saved = loadz(OUT / "outputs.npz")
    np.testing.assert_array_equal(saved["dates"], [str(day) for day in rebuilt["dates"]])
    np.testing.assert_array_equal(saved["currencies"], rebuilt["currencies"])
    same(saved["features"], rebuilt["matrix"])
    np.testing.assert_array_equal(saved["hard_quality"], rebuilt["hard"])
    same(saved["quality"], rebuilt["quality"])
    for h in HORIZONS:
        item = rebuilt["raw"][h]
        for name in ("target", "baseline", "candidate", "local"):
            same(saved[f"{name}__h{h}"], item[name])
        for name in ("active", "n_train"):
            np.testing.assert_array_equal(saved[f"{name}__h{h}"], item[name])
        np.testing.assert_array_equal(
            saved[f"maturity__h{h}"], [str(day) for day in item["maturity"]]
        )
        inactive = ~item["active"]
        np.testing.assert_array_equal(item["candidate"][inactive], item["baseline"][inactive])
        assert np.all(item["active"] <= rebuilt["hard"])

    stored_metrics = pd.read_csv(OUT / "metrics.csv")
    stored_gates = pd.read_csv(OUT / "historical_gates.csv")
    pd.testing.assert_frame_equal(stored_metrics, rebuilt["metrics"], check_dtype=False)
    pd.testing.assert_frame_equal(stored_gates, rebuilt["gates"], check_dtype=False)
    assert rebuilt["open_horizons"] == [3]

    logs = rebuilt["logs"]
    fitted = logs[logs.model_fit]
    for row in fitted.itertuples():
        origin = dt.date.fromisoformat(row.origin)
        maturity = dt.date.fromisoformat(row.last_target_maturity)
        assert maturity < origin - dt.timedelta(days=EMBARGO_DAYS)
    active_any = np.logical_or.reduce([
        rebuilt["raw"][h]["active"] for h in HORIZONS
    ])
    assert set(rebuilt["currencies"][active_any]) == {"AMD", "KZT"}

    # Corrupt every later feature, quality flag and target. Earlier predictions
    # must remain identical because quarterly heads use only mature past rows.
    boundary = dt.date(2025, 1, 1)
    future = rebuilt["dates"] >= boundary
    corrupted_matrix = rebuilt["matrix"].copy()
    corrupted_matrix[future] = corrupted_matrix[future] * -100.0 + 777.0
    corrupted_hard = rebuilt["hard"].copy()
    corrupted_hard[future] = ~corrupted_hard[future]
    corrupted_quality = rebuilt["quality"].copy()
    corrupted_quality[future] = 1.0 - corrupted_quality[future]
    for h in HORIZONS:
        item = rebuilt["raw"][h]
        target = item["target"].copy()
        finite = future & np.isfinite(target)
        target[finite] = 1.0 - target[finite]
        changed = fit_local_heads(
            corrupted_matrix, target, item["baseline"], item["maturity"],
            rebuilt["dates"], rebuilt["currencies"], corrupted_hard,
            corrupted_quality,
        )
        same(changed[0][~future], item["candidate"][~future])
        same(changed[1][~future], item["local"][~future])
        np.testing.assert_array_equal(changed[2][~future], item["active"][~future])

    checks = {
        "source_hashes_match": True,
        "features_and_quarterly_predictions_rebuilt": True,
        "historical_metrics_and_gates_rebuilt": True,
        "fallback_rows_equal_t5_exactly": True,
        "local_head_requires_hard_quality": True,
        "only_amd_kzt_heads_activated": True,
        "maturity_and_embargo_verified": True,
        "future_feature_label_quality_corruption_prefix_passed": True,
        "open_horizons": rebuilt["open_horizons"],
        "production_promoted": False,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
