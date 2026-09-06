import hashlib
import json
from pathlib import Path

import pandas as pd

from research.temperature_t19_anytime_quality_audit import OUT


def test_t19_saved_tables_cover_both_paths_all_clocks_and_horizons():
    metadata = json.loads((OUT / "metadata.json").read_text())
    weak = pd.read_csv(OUT / "weak_spots.csv")
    bootstrap = pd.read_csv(OUT / "paired_bootstrap.csv")
    assert metadata["selection_or_refit"] is False
    assert metadata["opened_period_diagnostic_only"] is True
    assert metadata["query_rows_per_scenario"] == 96700
    assert len(weak) == 200
    assert set(weak.scenario) == {
        "calendar_assumed_replay", "no_same_day_receipt"}
    assert set(weak.h) == {1, 3, 5, 10, 20}
    assert weak.clock.nunique() == 20
    assert len(bootstrap) == 800
    assert set(bootstrap.block_dates) == {20, 50}


def test_t19_hashes_and_limitations_are_explicit():
    metadata = json.loads((OUT / "metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert metadata["historical_receipts_certified"] is False
    assert metadata["bank_execution_validated"] is False


def test_t19_exposes_h20_and_local_calibration_weakness():
    weak = pd.read_csv(OUT / "weak_spots.csv")
    slices = pd.read_csv(OUT / "slice_weak_spots.csv")
    assert not weak[weak.h.eq(20)].probability_supported.any()
    assert not weak[weak.h.eq(20)].benefit_supported.any()
    currency_year = slices[slices.slice.eq("currency_year")]
    assert currency_year.flag_high_ece.any()
