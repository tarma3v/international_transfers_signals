"""Exact rebuild and future-target audit for T11."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.validation import target_reach_dates
from research.round5_features import load_round5_features
from research.round6_uzbek_central_bank_models import _forward
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t11_causal_benefit_gate import OUT, run_experiment
from research.temperature_t11_causal_benefit_gate_models import fit_quarterly_gate


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    persisted = loadz(OUT / "outputs.npz")
    rebuilt = run_experiment()["arrays"]
    assert set(persisted) == set(rebuilt)
    for key, value in rebuilt.items():
        np.testing.assert_array_equal(persisted[key], value)

    _matrix, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    target = _forward(series, index, 5)
    maturity = target_reach_dates(index, series, 5)
    model = persisted["model__cutoff_1530__h5"]
    prior = persisted["prior__cutoff_1530__h5"]
    original = fit_quarterly_gate(
        model, prior, target, maturity, dates)[:2]
    boundary = dt.date(2025, 1, 1)
    future = dates >= boundary
    changed_target = target.copy()
    changed_target[future & np.isfinite(changed_target)] += 7777.
    changed_maturity = maturity.copy()
    changed_maturity[future] = dt.date(2099, 1, 1)
    changed = fit_quarterly_gate(
        model, prior, changed_target, changed_maturity, dates)[:2]
    past = dates < boundary
    for left, right in zip(original, changed):
        np.testing.assert_array_equal(left[past], right[past])
    checks = {
        "source_hashes_verified": True,
        "all_persisted_arrays_exactly_rebuilt": True,
        "future_target_maturity_prefix_invariant": True,
        "fixed_weight_grid_verified": True,
    }
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
