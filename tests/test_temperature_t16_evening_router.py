import datetime as dt

import pandas as pd

from ml.transfer_temperature import MOSCOW, score_snapshot_as_of
from research.temperature_t16_evening_router import OUT


def test_t16_reports_separate_probability_and_benefit_sources():
    snapshots = pd.read_csv(OUT / "snapshots.csv.gz")
    query = dt.datetime(2026, 9, 1, 23, 15, tzinfo=MOSCOW)
    h3 = score_snapshot_as_of(snapshots, "KZT", query, 3)
    h10 = score_snapshot_as_of(snapshots, "KZT", query, 10)
    assert h3["source_kind"] == "post_receipt_market"
    assert h3["benefit_source_kind"] == "post_receipt_perpetual"
    assert h3["benefit_last_source_at"].endswith("22:59:59+03:00")
    assert h10["source_kind"] == "post_receipt_market"
    assert h10["benefit_source_kind"] == "post_receipt_market"
