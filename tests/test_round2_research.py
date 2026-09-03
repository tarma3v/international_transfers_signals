import datetime as dt

import numpy as np
import pandas as pd

from research.round2_diverse_models import _panel_features
from research.round2_external_data import _asof
from research.round2_router import EXPERTS, _past_oof


def test_panel_common_features_do_not_read_future_dates():
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(12)]
    index = [(currency, i, day) for currency in ("AAA", "BBB")
             for i, day in enumerate(days)]
    names = ["ret_1", "ret_5", "ret_20", "ret_60", "raw_vol_20"]
    rng = np.random.default_rng(42)
    original = rng.normal(size=(len(index), len(names)))
    changed = original.copy()
    cutoff = dt.date(2020, 1, 7)
    future = np.asarray([day > cutoff for _currency, _i, day in index])
    changed[future] = rng.normal(1000, 10, size=changed[future].shape)

    left, left_names = _panel_features(original, names, index)
    right, right_names = _panel_features(changed, names, index)
    past = ~future
    assert left_names == right_names
    np.testing.assert_allclose(left[past], right[past], rtol=0, atol=0)


def test_external_asof_never_uses_a_future_release():
    calendar = pd.DataFrame({"date": pd.to_datetime([
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"
    ])})
    source = pd.DataFrame({
        "available_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
        "value": [10.0, 20.0],
    })
    joined = _asof(calendar, source)
    assert np.isnan(joined.loc[0, "value"])
    assert joined.loc[1, "value"] == 10.0
    assert joined.loc[2, "value"] == 10.0
    assert joined.loc[3, "value"] == 20.0


def test_router_second_layer_leaves_a_full_calibration_year_gap():
    template = {
        year: {
            "test_idx": np.asarray([year], dtype=int),
            "test_score": np.asarray([year / 10000], dtype=float),
        }
        for year in (2017, 2018, 2019, 2020, 2022, 2023)
    }
    ranked = {expert: template for expert in EXPERTS}
    rows, scores = _past_oof(ranked, 2023)
    assert rows.tolist() == [2017, 2018, 2019, 2020]
    assert scores.shape == (4, len(EXPERTS))
    assert 2022 not in rows

