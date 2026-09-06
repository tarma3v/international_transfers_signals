import datetime as dt

import pandas as pd

from ml.transfer_temperature import MOSCOW, score_snapshot_as_of
from research.temperature_t14_perpetual_router import OUT


def test_t14_uses_horizon_specific_provenance_at_0900():
    snapshots = pd.read_csv(OUT / "snapshots.csv.gz")
    query = dt.datetime(2026, 9, 2, 9, 15, tzinfo=MOSCOW)
    h1 = score_snapshot_as_of(snapshots, "KZT", query, 1)
    h5 = score_snapshot_as_of(snapshots, "KZT", query, 5)
    assert h1["source_kind"] == "moex_perpetual_prefix"
    assert h1["last_source_at"].endswith("08:59:59+03:00")
    assert h1["confidence"] == "mature_history"
    assert h5["source_kind"] == "cbr_history"
    assert h5["last_source_at"].endswith("00:00:00+03:00")
    assert h5["confidence"] == "limited"
