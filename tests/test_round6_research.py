import csv
import datetime as dt
import pickle
from pathlib import Path

import numpy as np

from ml.data import CORRIDORS, Series
from research.round6_broad_cbr_features import build_broad_features
from research.round6_cny_basis_features import causality_check as basis_causality_check
from research.round6_cny_decomposition import delayed_by_currency
from research.round6_direct_rankers import _group_keys
from research.round6_moex_features import TICKERS, build_moex_features
from research.round6_moex_context_features import (
    TICKERS as CONTEXT_TICKERS,
    causality_check as context_causality_check,
)
from research.round6_cny_microstructure_features import (
    causality_check as microstructure_causality_check,
)
from research.round6_cny_hierarchical_logit import hierarchical_matrix
from research.round6_cny_lifecycle import YEARS as LIFECYCLE_YEARS, _stitch
from research.round6_cny_gam import N_CURRENCY, N_NUMERIC, gam_model
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_cny_rocket_features import (
    causality_check as rocket_causality_check,
)
from research.round6_cny_error_regime import row_scores as regime_row_scores
from research.round6_cny_trajectory_analogues import _scale as analogue_scale
from research.round6_cny_waveform_features import (
    causality_check as waveform_causality_check,
)
from research.round6_crossbank_consensus import (
    build_crossbank_features,
    causality_check as crossbank_causality_check,
)
from research.round6_crossbank_revision_dynamics import (
    causality_check as revision_causality_check,
)
from research.round6_crossbank_normalized_factor import (
    causality_check as normalized_factor_causality_check,
)
from research.round6_target_state_space import (
    causality_check as target_state_causality_check,
)
from research.round6_belarus_nbrb_features import (
    causality_check as nbrb_causality_check,
    load_nbrb,
)
from research.round6_georgia_nbg_features import (
    causality_check as nbg_causality_check,
    load_nbg,
)
from research.round6_multiobjective_blend import combine_causal
from research.round6_multihorizon_policy_screen import (
    exponential_fired,
)
from research.round6_refit_score_normalization import causal_normalizers
from research.round6_refit_threshold import ThresholdPolicy, quarter_reset_fired
from research.round6_resolved_features import build_resolved_features
from research.round6_rate_control import Policy, controlled_fired
from research.round6_shared_horizon import barrier_targets
from research.round6_verify_freeze import verify as verify_prospective_freeze
from research.round6_weekly_confidence_policy import (
    WeeklyPolicy,
    weekly_fired,
)


def _panel_output():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(50)]
    rows = [(currency, day) for currency in CORRIDORS for day in days]
    dates = np.asarray([day for _currency, day in rows], dtype=object)
    currencies = np.asarray([currency for currency, _day in rows], dtype=object)
    scores = np.asarray([
        np.sin(i / 4.0) + .01 * number
        for number, _currency in enumerate(CORRIDORS)
        for i, _day in enumerate(days)
    ])
    output = {
        2025: {
            "calib_idx": np.asarray([], dtype=int),
            "test_idx": np.arange(len(rows)),
            "calib_score": np.asarray([]),
            "test_score": scores,
        }
    }
    return output, dates, currencies, np.ones(len(rows))


def test_weekly_controller_cannot_change_past_decisions_from_future_scores():
    output, dates, currencies, y = _panel_output()
    policy = Policy(20, .80, .30, 3, 2)
    _, first, _ = controlled_fired(output, (2025,), dates, currencies, y, policy)
    cutoff = dt.date(2025, 1, 25)

    changed = {2025: {key: value.copy() for key, value in output[2025].items()}}
    future = dates > cutoff
    changed[2025]["test_score"][future] = np.linspace(-1000, 1000, int(future.sum()))
    _, second, _ = controlled_fired(changed, (2025,), dates, currencies, y, policy)
    np.testing.assert_array_equal(first[dates <= cutoff], second[dates <= cutoff])


def test_weekly_controller_respects_per_currency_cap():
    output, dates, currencies, y = _panel_output()
    policy = Policy(10, .50, .00, 3, 2)
    valid, fired, _ = controlled_fired(output, (2025,), dates, currencies, y, policy)
    for currency in CORRIDORS:
        weeks = {}
        for day in dates[valid & fired & (currencies == currency)]:
            key = tuple(day.isocalendar()[:2])
            weeks[key] = weeks.get(key, 0) + 1
        assert max(weeks.values()) <= policy.weekly_cap


def test_new_external_archives_have_full_history_and_expected_units():
    nbg, _nbg_digest = load_nbg()
    nbrb, _nbrb_digest = load_nbrb()
    assert all(len(nbg[code]) > 3000 for code in ("RUB", "USD", "CNY"))
    assert all(len(nbrb[code]) == 3901 for code in ("RUB", "USD", "CNY"))
    # Both archives are normalized to local currency per one quoted unit.
    assert .01 < nbg["RUB"].values[-1] < .10
    assert .01 < nbrb["RUB"].values[-1] < .10


def test_new_external_cross_features_ignore_future_values():
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)
    ], dtype=object)
    local = {
        "RUB": Series("RUB", days.copy(), .03 + np.arange(40) * 1e-5),
        "USD": Series("USD", days.copy(), 3.0 + np.arange(40) * 1e-3),
        "CNY": Series("CNY", days.copy(), .42 + np.arange(40) * 1e-4),
    }
    references = {
        "USD": Series("USD", days.copy(), 90.0 + np.arange(40) * .02),
        "CNY": Series("CNY", days.copy(), 12.5 + np.arange(40) * .005),
    }
    index = [(CORRIDORS[0], i, day) for i, day in enumerate(days[6:36])]
    cutoff = days[25]
    assert nbg_causality_check(index, references, local, cutoff=cutoff)
    assert nbrb_causality_check(index, references, local, cutoff=cutoff)


def test_crossbank_consensus_cancels_each_sources_domestic_unit():
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(20)
    ], dtype=object)
    base = {
        "RUB": Series("RUB", days.copy(), .03 + np.arange(20) * 1e-5),
        "USD": Series("USD", days.copy(), 3.0 + np.arange(20) * 1e-3),
        "CNY": Series("CNY", days.copy(), .42 + np.arange(20) * 1e-4),
    }
    references = {
        "USD": Series("USD", days.copy(), 90.0 + np.arange(20) * .02),
        "CNY": Series("CNY", days.copy(), 12.5 + np.arange(20) * .005),
    }
    index = [(CORRIDORS[0], i, day) for i, day in enumerate(days[3:18])]
    sources = {
        f"source_{number}": {
            code: Series(code, series.dates.copy(), series.values * scale)
            for code, series in base.items()
        }
        for number, scale in enumerate((1.0, 10.0, 1000.0))
    }
    matrix, names, _ = build_crossbank_features(index, references, sources)
    reference, reference_names, _ = build_crossbank_features(
        index, references, {f"source_{number}": base for number in range(3)},
    )
    assert names == reference_names
    np.testing.assert_allclose(matrix, reference, rtol=0, atol=1e-3)


def test_crossbank_consensus_ignores_future_local_cb_values():
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)
    ], dtype=object)
    references = {
        "USD": Series("USD", days.copy(), 90.0 + np.arange(40) * .02),
        "CNY": Series("CNY", days.copy(), 12.5 + np.arange(40) * .005),
    }
    sources = {}
    for source_name, scale in zip(
        ("armenia_cba", "source_1", "source_2"), (1.0, 4.0, 12.0),
    ):
        sources[source_name] = {
            "RUB": Series("RUB", days.copy(), scale * (.03 + np.arange(40) * 1e-5)),
            "USD": Series("USD", days.copy(), scale * (3.0 + np.arange(40) * 1e-3)),
            "CNY": Series("CNY", days.copy(), scale * (.42 + np.arange(40) * 1e-4)),
        }
    index = [(CORRIDORS[0], i, day) for i, day in enumerate(days[6:36])]
    assert crossbank_causality_check(
        index, references, sources, cutoff=days[25],
    )


def test_joint_external_refits_use_only_resolved_labels():
    path = Path("results/research/round6/joint_external_stack/training_log.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        quarter = dt.date.fromisoformat(row["quarter"])
        last_resolved = dt.date.fromisoformat(row["last_resolved"])
        assert last_resolved < quarter
        assert int(row["n_train"]) >= 1000


def test_crossbank_revision_features_ignore_future_local_cb_values():
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)
    ], dtype=object)
    references = {
        "USD": Series("USD", days.copy(), 90.0 + np.arange(40) * .02),
        "CNY": Series("CNY", days.copy(), 12.5 + np.arange(40) * .005),
    }
    sources = {}
    for number, scale in enumerate((1.0, 4.0, 12.0)):
        sources[f"source_{number}"] = {
            "RUB": Series("RUB", days.copy(), scale * (.03 + np.arange(40) * 1e-5)),
            "USD": Series("USD", days.copy(), scale * (3.0 + np.arange(40) * 1e-3)),
            "CNY": Series("CNY", days.copy(), scale * (.42 + np.arange(40) * 1e-4)),
        }
    index = [(CORRIDORS[0], i, day) for i, day in enumerate(days[6:36])]
    assert revision_causality_check(
        index, references, sources, cutoff=days[25],
    )


def test_normalized_crossbank_factor_ignores_future_local_cb_values():
    days = np.asarray([
        dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(100)
    ], dtype=object)
    references = {
        "USD": Series("USD", days.copy(), 90.0 + np.arange(100) * .02),
        "CNY": Series("CNY", days.copy(), 12.5 + np.arange(100) * .005),
    }
    sources = {}
    for source_name, scale in zip(
        ("armenia_cba", "source_1", "source_2"), (1.0, 4.0, 12.0),
    ):
        sources[source_name] = {
            "RUB": Series("RUB", days.copy(), scale * (.03 + np.arange(100) * 1e-5)),
            "USD": Series("USD", days.copy(), scale * (3.0 + np.arange(100) * 1e-3)),
            "CNY": Series("CNY", days.copy(), scale * (.42 + np.arange(100) * 1e-4)),
        }
    index = [(CORRIDORS[0], i, day) for i, day in enumerate(days[6:96])]
    assert normalized_factor_causality_check(
        index, references, sources, cutoff=days[80],
    )


def test_target_state_space_ignores_future_target_values():
    length = 100
    days = np.asarray([
        dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(length)
    ], dtype=object)
    series = {}
    for number, code in enumerate(CORRIDORS):
        values = (1.0 + number) * np.exp(
            .001 * np.arange(length) + .01 * np.sin(np.arange(length) / 7.0)
        )
        series[code] = Series(code, days.copy(), values)
    index = [
        (code, position, days[position])
        for code in CORRIDORS for position in range(10, 95)
    ]
    assert target_state_causality_check(
        series, index, cutoff=days[75],
    )


def test_exponential_threshold_cannot_change_past_from_future_scores():
    output, dates, currencies, y = _panel_output()
    _, first = exponential_fired(
        output, (2025,), dates, currencies, y, rate=.25, half_life=20,
    )
    cutoff = dt.date(2025, 1, 25)
    changed = {2025: {key: value.copy() for key, value in output[2025].items()}}
    future = dates > cutoff
    changed[2025]["test_score"][future] = np.linspace(-1000, 1000, int(future.sum()))
    _, second = exponential_fired(
        changed, (2025,), dates, currencies, y, rate=.25, half_life=20,
    )
    np.testing.assert_array_equal(first[dates <= cutoff], second[dates <= cutoff])


def test_weekly_confidence_policy_is_causal_and_respects_cap():
    output, dates, currencies, y = _panel_output()
    policy = WeeklyPolicy(20, .70, .90, .50, 3, 2)
    valid, first, _ = weekly_fired(
        output, (2025,), dates, currencies, y, policy,
    )
    cutoff = dt.date(2025, 1, 25)
    changed = {2025: {key: value.copy() for key, value in output[2025].items()}}
    future = dates > cutoff
    changed[2025]["test_score"][future] = np.linspace(-1000, 1000, int(future.sum()))
    _, second, _ = weekly_fired(
        changed, (2025,), dates, currencies, y, policy,
    )
    np.testing.assert_array_equal(first[dates <= cutoff], second[dates <= cutoff])
    for currency in CORRIDORS:
        counts = {}
        for day in dates[valid & first & (currencies == currency)]:
            week = tuple(day.isocalendar()[:2])
            counts[week] = counts.get(week, 0) + 1
        assert max(counts.values(), default=0) <= policy.cap


def test_resolved_outcome_features_ignore_unresolved_future():
    length = 80
    days = np.asarray([
        dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(length)
    ], dtype=object)
    series = {}
    for number, code in enumerate(CORRIDORS):
        returns = .002 * np.sin(np.arange(length - 1) / (3.0 + number))
        values = (1.0 + number) * np.exp(np.r_[0.0, np.cumsum(returns)])
        series[code] = Series(code, days.copy(), values)
    index = [
        (code, position, days[position])
        for code in CORRIDORS for position in range(20, length)
    ]
    full, names = build_resolved_features(series, index)
    cutoff = dt.date(2020, 2, 25)
    corrupted = {}
    for code, item in series.items():
        values = item.values.copy()
        mask = item.dates > cutoff
        values[mask] *= np.linspace(2.0, 20.0, int(mask.sum()))
        corrupted[code] = Series(code, item.dates.copy(), values)
    changed, changed_names = build_resolved_features(corrupted, index)
    past = np.asarray([row[2] <= cutoff for row in index])
    assert names == changed_names
    np.testing.assert_array_equal(full[past], changed[past])
    assert not np.array_equal(full[~past], changed[~past])


def test_broad_cbr_features_ignore_future_reference_values():
    length = 90
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(length)
    ], dtype=object)
    targets = {
        code: Series(code, days.copy(), np.exp(np.arange(length) * .001 + number))
        for number, code in enumerate(CORRIDORS)
    }
    references = {
        code: Series(code, days.copy(), np.exp(np.arange(length) * scale))
        for code, scale in (("USD", .001), ("EUR", .0012), ("AUD", .0008))
    }
    index = [
        (code, position, days[position])
        for code in CORRIDORS for position in range(20, length)
    ]
    full, names = build_broad_features(references, targets, index)
    cutoff = dt.date(2025, 2, 25)
    changed = {}
    for code, item in references.items():
        values = item.values.copy()
        future = item.dates > cutoff
        values[future] *= np.linspace(2.0, 30.0, int(future.sum()))
        changed[code] = Series(code, item.dates.copy(), values)
    corrupted, corrupted_names = build_broad_features(changed, targets, index)
    past = np.asarray([row[2] <= cutoff for row in index])
    assert names == corrupted_names
    np.testing.assert_array_equal(full[past], corrupted[past])
    assert np.any(full[~past] != corrupted[~past])


def test_quarter_reset_threshold_cannot_change_past_from_future_scores():
    output, dates, currencies, y = _panel_output()
    policy = ThresholdPolicy(.25, 20, 5)
    _, first, _ = quarter_reset_fired(
        output, (2025,), dates, currencies, y, policy,
    )
    cutoff = dt.date(2025, 1, 25)
    changed = {2025: {key: value.copy() for key, value in output[2025].items()}}
    future = dates > cutoff
    changed[2025]["test_score"][future] = np.linspace(-500, 500, int(future.sum()))
    _, second, _ = quarter_reset_fired(
        changed, (2025,), dates, currencies, y, policy,
    )
    np.testing.assert_array_equal(first[dates <= cutoff], second[dates <= cutoff])


def test_causal_blend_handles_missing_prior_calibration_without_future_rank():
    dates = np.asarray([
        dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(12)
    ], dtype=object)
    currencies = np.asarray([CORRIDORS[0]] * len(dates), dtype=object)
    test = np.arange(6, 12)
    empty_part = {2024: {
        "calib_idx": np.asarray([], dtype=int),
        "test_idx": test,
        "calib_score": np.asarray([], dtype=float),
        "test_score": np.arange(6, dtype=float),
    }}
    calibrated_part = {2024: {
        "calib_idx": np.arange(6),
        "test_idx": test,
        "calib_score": np.arange(6, dtype=float),
        "test_score": np.arange(6, 12, dtype=float),
    }}
    first = combine_causal(
        [empty_part, calibrated_part], (.5, .5), dates, currencies,
    )[2024]["test_score"]
    changed_part = {2024: {
        key: value.copy() for key, value in empty_part[2024].items()
    }}
    changed_part[2024]["test_score"][-1] = -1000.0
    second = combine_causal(
        [changed_part, calibrated_part], (.5, .5), dates, currencies,
    )[2024]["test_score"]
    np.testing.assert_array_equal(first[:-1], second[:-1])


def test_refit_score_normalizers_cannot_propagate_future_scores_backward():
    output, dates, currencies, _y = _panel_output()
    raw = output[2025]["test_score"].copy()
    first_quarter, first_day = causal_normalizers(raw, dates, currencies, 5)
    cutoff = dt.date(2025, 1, 25)
    changed = raw.copy()
    future = dates > cutoff
    changed[future] = np.linspace(-1000, 1000, int(future.sum()))
    second_quarter, second_day = causal_normalizers(changed, dates, currencies, 5)
    np.testing.assert_array_equal(first_quarter[~future], second_quarter[~future])
    np.testing.assert_array_equal(first_day[~future], second_day[~future])


def test_week_query_key_uses_iso_year_at_calendar_boundary():
    dates = np.asarray([dt.date(2024, 12, 30), dt.date(2025, 1, 2)], dtype=object)
    currencies = np.asarray([CORRIDORS[0], CORRIDORS[0]], dtype=object)
    keys = _group_keys(dates, currencies, "week")
    assert keys[0] == keys[1]
    assert keys[0].endswith("2025W01")


def test_shared_horizon_barriers_match_each_future_publication():
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(7)
    ], dtype=object)
    series = {CORRIDORS[0]: Series(
        CORRIDORS[0], days, np.asarray([2.0, 3.0, 1.0, 2.0, 4.0, 2.0, 5.0]),
    )}
    index = [(CORRIDORS[0], position, days[position]) for position in range(7)]
    barriers = barrier_targets(index, series)
    np.testing.assert_array_equal(
        barriers[0], np.asarray([1.0, 0.0, 1.0, 1.0, 1.0]),
    )
    assert np.isnan(barriers[2:]).all()


def _market_row(day, close):
    return {
        "date": day, "open": close - .2, "high": close + .3,
        "low": close - .4, "close": close, "waprice": close - .1,
        "trades": 100.0 + close,
    }


def test_moex_features_exclude_same_day_and_future_closes():
    days = [dt.date(2025, 1, day) for day in (1, 2, 3, 4)]
    history = {
        ticker: [_market_row(day, 100.0 + i) for i, day in enumerate(days)]
        for ticker in TICKERS
    }
    index = [(CORRIDORS[0], 0, days[2])]
    original, names = build_moex_features(index, history)
    changed = {ticker: [dict(row) for row in rows] for ticker, rows in history.items()}
    for ticker in TICKERS:
        for row in changed[ticker]:
            if row["date"] >= days[2]:
                row["close"] *= 100.0
                row["open"] *= 100.0
                row["high"] *= 100.0
                row["low"] *= 100.0
                row["waprice"] *= 100.0
                row["trades"] *= 100.0
    corrupted, corrupted_names = build_moex_features(index, changed)
    assert names == corrupted_names
    np.testing.assert_array_equal(original, corrupted)


def test_currency_delay_never_propagates_future_rows_backward():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(6)]
    index = [
        (currency, position, day)
        for currency in CORRIDORS[:2]
        for position, day in enumerate(days)
    ]
    matrix = np.arange(len(index) * 2, dtype=float).reshape(len(index), 2)
    first = delayed_by_currency(matrix, index, rows=2)
    changed = matrix.copy()
    changed[[4, 5, 10, 11]] += 10000.0
    second = delayed_by_currency(changed, index, rows=2)
    np.testing.assert_array_equal(first[:6], second[:6])
    np.testing.assert_array_equal(first[6:], second[6:])
    np.testing.assert_array_equal(first[[0, 1, 6, 7]], 0.0)


def test_cny_basis_joint_market_and_cbr_asof_corruption():
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(25)
    ], dtype=object)
    history = {
        "CNYRUB_TOM": [
            _market_row(day, 10.0 + .01 * i) for i, day in enumerate(days)
        ]
    }
    cbr = Series("CNY", days.copy(), 9.9 + .008 * np.arange(len(days)))
    index = [
        (CORRIDORS[0], i, day) for i, day in enumerate(days[5:20])
    ]
    assert basis_causality_check(
        index, history, cbr, cutoff=dt.date(2025, 1, 14),
    )


def test_moex_context_features_ignore_same_day_and_future_rows():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(25)]
    history = {}
    for ticker_number, ticker in enumerate(CONTEXT_TICKERS):
        rows = []
        for i, day in enumerate(days):
            close = 100.0 + ticker_number * 20.0 + i
            rows.append({
                "date": day,
                "open": close - .2,
                "high": close + .3,
                "low": close - .4,
                "close": close,
                "activity": 1000.0 + 10.0 * i,
                "yield": 7.0 + .01 * i,
            })
        history[ticker] = rows
    index = [
        (CORRIDORS[0], i, day) for i, day in enumerate(days[5:20])
    ]
    assert context_causality_check(
        index, history, cutoff=dt.date(2025, 1, 14),
    )


def test_cny_microstructure_ignores_same_day_and_future_rows():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(25)]
    history = {
        "CNYRUB_TOM": [
            _market_row(day, 10.0 + .01 * i) for i, day in enumerate(days)
        ]
    }
    index = [
        (CORRIDORS[0], i, day) for i, day in enumerate(days[5:20])
    ]
    assert microstructure_causality_check(
        index, history, cutoff=dt.date(2025, 1, 14),
    )


def test_hierarchical_logit_interactions_are_currency_local():
    transparent = [
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_5", "ret_20",
    ]
    currency_names = [f"currency_{code}" for code in CORRIDORS]
    names = transparent + currency_names
    X = np.zeros((2, len(names)), dtype=float)
    X[:, :len(transparent)] = 1.0
    X[0, len(transparent)] = 1.0
    X[1, len(transparent) + 1] = 1.0
    moex_names = [
        "moex_cnyrub_tom_open_close",
        "moex_cnyrub_tom_intraday_range",
        "moex_cnyrub_tom_close_wap",
        "moex_cnyrub_tom_overnight_gap",
        "moex_cnyrub_tom_log_trades",
        "moex_cnyrub_tom_open_close_z_20",
        "moex_cnyrub_tom_intraday_range_z_20",
        "moex_cnyrub_tom_close_wap_z_20",
    ]
    moex = np.ones((2, len(moex_names)), dtype=float)
    matrix, feature_names = hierarchical_matrix(X, names, moex, moex_names)
    assert matrix.shape == (2, 89)
    assert len(feature_names) == 89
    interaction = matrix[:, 19:].reshape(2, len(CORRIDORS), 14)
    np.testing.assert_array_equal(interaction[0, 0], 1.0)
    np.testing.assert_array_equal(interaction[0, 1:], 0.0)
    np.testing.assert_array_equal(interaction[1, 1], 1.0)
    np.testing.assert_array_equal(interaction[1, [0, 2, 3, 4]], 0.0)


def test_prospective_freeze_inputs_are_byte_identical():
    assert verify_prospective_freeze() == []


def test_lifecycle_stitch_requires_each_year_exactly_once():
    parts = [{year: {"test_score": np.asarray([year])}} for year in LIFECYCLE_YEARS]
    stitched = _stitch(*parts)
    assert tuple(stitched) == LIFECYCLE_YEARS
    try:
        _stitch(parts[0], parts[0], *parts[1:])
    except AssertionError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("duplicate lifecycle year was silently accepted")


def test_shock_weight_variants_change_only_2022_and_2023():
    path = Path(
        "results/research/round6/cny_shock_weight_plateau/lifecycle_outputs.pkl"
    )
    with path.open("rb") as handle:
        outputs = pickle.load(handle)
    control = outputs["cny100_anchor000"]
    assert tuple(sorted(control)) == LIFECYCLE_YEARS
    for candidate, output in outputs.items():
        if candidate == "cny100_anchor000":
            continue
        for year in set(LIFECYCLE_YEARS) - {2022, 2023}:
            np.testing.assert_array_equal(
                output[year]["test_idx"], control[year]["test_idx"],
            )
            np.testing.assert_array_equal(
                output[year]["test_score"], control[year]["test_score"],
            )


def test_gam_pipelines_fit_fixed_raw_schema_and_return_probabilities():
    rng = np.random.default_rng(20260905)
    matrix = rng.normal(size=(80, N_NUMERIC + N_CURRENCY))
    matrix[:, N_NUMERIC:] = 0.0
    matrix[np.arange(len(matrix)), N_NUMERIC + np.arange(len(matrix)) % N_CURRENCY] = 1.0
    target = (np.arange(len(matrix)) % 3 == 0).astype(float)
    for kind in ("market", "all"):
        probabilities = gam_model(kind).fit(matrix[:60], target[:60]).predict_proba(
            matrix[60:]
        )[:, 1]
        assert probabilities.shape == (20,)
        assert np.all(np.isfinite(probabilities))
        assert np.all((probabilities > 0) & (probabilities < 1))


def test_reliability_percentiles_cannot_propagate_future_scores_backward():
    days = np.asarray([
        dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(50)
    ], dtype=object)
    dates = np.tile(days, 2)
    currencies = np.repeat(CORRIDORS[:2], len(days))
    scores = np.r_[np.sin(np.arange(50) / 5), np.cos(np.arange(50) / 6)]
    first = causal_percentiles(scores, dates, currencies, window=20, minimum=5)
    changed = scores.copy()
    changed[(dates >= days[40])] = np.linspace(-1000, 1000, np.sum(dates >= days[40]))
    second = causal_percentiles(changed, dates, currencies, window=20, minimum=5)
    np.testing.assert_array_equal(first[dates < days[40]], second[dates < days[40]])


def test_analogue_scaling_is_fit_only_on_training_rows():
    train = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    test = np.asarray([[4.0, 9.0], [5.0, 11.0]])
    scaled_train, scaled_test = analogue_scale(train, test)
    changed_train, changed_test = analogue_scale(train, test * 1000.0)
    np.testing.assert_array_equal(scaled_train, changed_train)
    assert not np.array_equal(scaled_test, changed_test)


def test_cny_waveform_ignores_same_day_and_future_sessions():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(35)]
    history = {
        "CNYRUB_TOM": [
            _market_row(day, 10.0 + .01 * i + .002 * np.sin(i))
            for i, day in enumerate(days)
        ]
    }
    index = [
        (CORRIDORS[0], i, day) for i, day in enumerate(days[21:32])
    ]
    assert waveform_causality_check(
        index, history, cutoff=days[27],
    )


def test_cny_random_convolutions_ignore_same_day_and_future_sessions():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(35)]
    history = {
        "CNYRUB_TOM": [
            _market_row(day, 10.0 + .01 * i + .002 * np.sin(i))
            for i, day in enumerate(days)
        ]
    }
    index = [
        (CORRIDORS[0], i, day) for i, day in enumerate(days[21:32])
    ]
    assert rocket_causality_check(
        index, history, cutoff=days[27],
    )


def test_regime_row_scores_prefer_each_rows_own_year_test_score():
    output = {
        2024: {
            "calib_idx": np.asarray([0]),
            "calib_score": np.asarray([.1]),
            "test_idx": np.asarray([1]),
            "test_score": np.asarray([.2]),
        },
        2025: {
            "calib_idx": np.asarray([1]),
            "calib_score": np.asarray([.9]),
            "test_idx": np.asarray([2]),
            "test_score": np.asarray([.3]),
        },
    }
    np.testing.assert_array_equal(
        regime_row_scores(output, 3), np.asarray([.1, .2, .3]),
    )


def test_regime_lifecycle_handoff_changes_only_2024_and_later():
    lifecycle_path = Path(
        "results/research/round6/cny_rocket_lifecycle/outputs.pkl"
    )
    with lifecycle_path.open("rb") as handle:
        lifecycle = pickle.load(handle)
    control = lifecycle["primary_resolved2000"]
    challenger = lifecycle["primary_then_regime2024"]
    assert tuple(sorted(control)) == LIFECYCLE_YEARS
    assert tuple(sorted(challenger)) == LIFECYCLE_YEARS
    for year in range(2017, 2024):
        np.testing.assert_array_equal(
            challenger[year]["test_idx"], control[year]["test_idx"],
        )
        np.testing.assert_array_equal(
            challenger[year]["test_score"], control[year]["test_score"],
        )
