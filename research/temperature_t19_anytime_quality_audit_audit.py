"""Independent rebuild audit for the T19 unified quality tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from research.temperature_t19_anytime_quality_audit import OUT, run_audit


FILES = {
    "probability_metrics": "probability_metrics.csv",
    "benefit_metrics": "benefit_metrics.csv",
    "reliability": "reliability_bins.csv",
    "benefit_bins": "benefit_calibration_bins.csv",
    "bootstrap": "paired_bootstrap.csv",
    "coverage": "coverage.csv",
    "weak": "weak_spots.csv",
    "slice_weak": "slice_weak_spots.csv",
    "sample": "selection_trace_sample.csv",
}


def _canonical(frame):
    frame = frame.copy()
    for column in frame.columns:
        if (column.endswith("_at") or column in {
                "query_date", "valid_from", "source_at",
                "benefit_source_at"}):
            frame[column] = frame[column].astype(str)
    columns = sorted(frame.columns)
    frame = frame[columns]
    sort = [column for column in (
        "scenario", "clock", "h", "slice", "group", "currency", "year",
        "weekday", "bin", "block_dates", "query_id",
    ) if column in frame]
    if sort:
        frame = frame.sort_values(sort).reset_index(drop=True)
    frame = frame.astype(object)
    return frame.where(pd.notna(frame), "__NULL__")


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest

    rebuilt = run_audit()
    for key, filename in FILES.items():
        saved = pd.read_csv(OUT / filename)
        pd.testing.assert_frame_equal(
            _canonical(saved), _canonical(rebuilt[key]),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )

    probability = rebuilt["probability_metrics"]
    benefit = rebuilt["benefit_metrics"]
    weak = rebuilt["weak"]
    bootstrap = rebuilt["bootstrap"]
    assert len(probability) == 4800
    assert len(benefit) == 4800
    assert len(weak) == 200
    assert len(bootstrap) == 800
    assert set(weak.h) == {1, 3, 5, 10, 20}
    assert set(weak.scenario) == {
        "calendar_assumed_replay", "no_same_day_receipt"}
    assert set(bootstrap.block_dates) == {20, 50}
    assert weak.ece.max() < .08
    assert not weak[weak.h.eq(20)].probability_supported.any()
    assert not weak[weak.h.eq(20)].benefit_supported.any()

    checks = {
        **rebuilt["checks"],
        "source_hashes_verified": True,
        "all_saved_tables_exactly_rebuilt": True,
        "probability_metric_rows": int(len(probability)),
        "benefit_metric_rows": int(len(benefit)),
        "paired_bootstrap_rows": int(len(bootstrap)),
        "state_horizon_rows": int(len(weak)),
        "overall_ece_below_008": True,
        "h20_not_supported_by_both_blocks": True,
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
