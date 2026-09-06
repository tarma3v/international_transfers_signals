import datetime as dt

import numpy as np

from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.temperature_t9_early_market_models import (
    CLOCKS,
    build_early_market_features,
    physical_causality_check,
)


def test_early_market_features_have_fixed_schema_and_causal_prefix():
    _matrix, _names, index, series, *_ = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, _digest = load_spot_1530_history()
    features, names, available, source_at = build_early_market_features(
        index, history, references, dt.time(9, 0))
    assert features.shape == (len(index), 32)
    assert len(names) == 32
    assert np.all(np.isfinite(features))
    assert available.dtype == bool and available.any()
    for i in np.flatnonzero(available):
        assert source_at[i] < dt.datetime.combine(index[i][2], dt.time(9, 0))
    assert physical_causality_check(index, history, references, CLOCKS[3])
