"""Тесты на корректность величин, а не только на честность.

Тесты честности (test_leakage.py) доказывают, что мы не заглядываем в будущее.
Они ничего не говорят о том, считают ли признаки и политика то, что обещают.
Здесь — проверки на входах с известным ответом.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from ml.data import Series
from ml.features import (
    RANGE_WINDOWS,
    RETURN_LAGS,
    VOL_WINDOWS,
    WARMUP,
    _bps,
    _cyclic,
    _position_in_range,
    _share_of_days_beaten,
    _up_share,
    row_features,
)
from ml.two_metrics import build_windows, evaluate_policy, oracle_gain, payday_anchors


def test_bps_sign_is_client_side() -> None:
    """Рост курса ЦБ — хуже клиенту, поэтому знак инвертирован."""
    assert _bps(110.0, 100.0) == pytest.approx(-1000.0)
    assert _bps(90.0, 100.0) == pytest.approx(+1000.0)
    assert _bps(100.0, 100.0) == 0.0


def test_position_in_range_hits_both_ends() -> None:
    assert _position_in_range(np.array([1.0, 2.0, 3.0]), 3) == pytest.approx(100.0)
    assert _position_in_range(np.array([3.0, 2.0, 1.0]), 3) == pytest.approx(0.0)
    assert _position_in_range(np.array([1.0, 3.0, 2.0]), 3) == pytest.approx(50.0)
    assert _position_in_range(np.array([5.0, 5.0, 5.0]), 3) == 50.0  # плоское окно


def test_share_of_days_beaten_is_a_real_percentile_in_the_client_direction() -> None:
    """Направление важнее формулы: низкий курс выгоден клиенту.

    На входе [1,2,3,4,10] сегодня — самый дорогой день окна, поэтому величина,
    стоящая за текстом «выгоднее, чем в X % дней», обязана быть 0, а не 100.
    """
    assert _share_of_days_beaten(np.array([1.0, 2.0, 3.0, 4.0, 10.0]), 5) == pytest.approx(0.0)
    assert _share_of_days_beaten(np.array([10.0, 2.0, 3.0, 4.0, 1.0]), 5) == pytest.approx(100.0)
    assert _share_of_days_beaten(np.array([1.0, 2.0, 3.0, 4.0, 2.5]), 5) == pytest.approx(50.0)


def test_position_in_range_and_percentile_disagree_on_an_outlier() -> None:
    """Именно поэтому у них разные имена и разные признаки.

    Один старый выброс наверху делает положение в диапазоне низким (сегодня
    почти на дне), тогда как побито всего 20 % дней. Текст «выгоднее, чем
    в 85 % дней» обещает вторую величину, а индикатор ТЗ «уровень» работает
    на первой — на этом входе они расходятся на 80 пунктов.
    """
    past = np.array([1.0] * 8 + [100.0] + [2.0])
    assert _position_in_range(past, 10) < 2.0
    assert _share_of_days_beaten(past, 10) <= 20.0


def test_up_share_bounds() -> None:
    assert _up_share(np.arange(1.0, 20.0)) == pytest.approx(100.0)
    assert _up_share(np.arange(20.0, 1.0, -1.0)) == pytest.approx(0.0)


def test_calendar_features_are_cyclic_pairs_not_ordinal_numbers() -> None:
    """Граница цикла не должна выглядеть для модели большим числовым скачком."""
    past = np.arange(WARMUP, dtype=float) + 100.0
    f = row_features(past, dt.date(2024, 12, 31), "TJS", 1, {}, None)

    for raw in ("dow", "dom", "month", "week_of_month", "quarter"):
        assert raw not in f
        assert f"{raw}_sin" in f and f"{raw}_cos" in f
        assert f[f"{raw}_sin"] ** 2 + f[f"{raw}_cos"] ** 2 == pytest.approx(1.0)

    assert _cyclic(0.0, 7.0) == pytest.approx((0.0, 1.0))


def test_streaks_count_consecutive_moves() -> None:
    past = np.array([10.0, 9.0, 8.0, 7.0])  # три падения подряд
    f = row_features(past, dt.date(2024, 3, 5), "TJS", 1, {}, None)
    assert f["streak_dn"] == 3.0
    assert f["streak_up"] == 0.0


def test_payday_anchors_land_on_paydays() -> None:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=k) for k in range(120)]
    for i in payday_anchors(dates):
        assert dates[i].day in (5, 6, 7, 20, 21, 22), dates[i]


def test_payday_anchors_are_monthly_and_ordered() -> None:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=k) for k in range(365)]
    idx = payday_anchors(dates)
    assert idx == sorted(idx)
    assert 20 <= len(idx) <= 26  # два в месяц за год, с поправкой на выходные


def test_evaluate_policy_takes_the_first_fire() -> None:
    """Политика обязана брать ПЕРВОЕ срабатывание окна, а не лучшее."""
    v = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    windows = [(0, 5)]
    run = evaluate_policy(v, windows, {2: True, 4: True})
    assert run.day_used.tolist() == [2] and int(run.fired.sum()) == 1
    assert run.mean == pytest.approx(_bps(98.0, 100.0))


def test_evaluate_policy_falls_back_to_window_end() -> None:
    v = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    run = evaluate_policy(v, [(0, 5)], {})
    assert run.fire_rate == 0.0 and run.day_used.tolist() == [5]
    assert run.mean == pytest.approx(_bps(95.0, 100.0))


def test_oracle_is_an_upper_bound_on_any_policy() -> None:
    rng = np.random.default_rng(7)
    v = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    windows = [(p, 5) for p in range(0, 180, 12)]
    orc = oracle_gain(v, windows)
    for seed in range(5):
        r = np.random.default_rng(seed)
        fires = {i: bool(r.integers(0, 2)) for i in range(len(v))}
        assert evaluate_policy(v, windows, fires).mean <= orc + 1e-9


def test_build_windows_stays_inside_the_series() -> None:
    dates = np.array([dt.date(2024, 1, 1) + dt.timedelta(days=k) for k in range(90)],
                     dtype=object)
    s = Series("TJS", dates, np.linspace(10.0, 11.0, 90))
    windows = build_windows({"TJS": s}, ("TJS",), 5, warmup=10)
    assert windows, "на 90 днях должно найтись хотя бы одно зарплатное окно"
    for (c, i), (p, remaining) in windows.items():
        assert c == "TJS"
        assert 10 <= p and p + 5 < len(s)
        assert i == p + (5 - remaining)


def test_share_above_sma_averages_exactly_w_points_ending_at_each_point() -> None:
    """`share_above_sma_w` — доля точек окна, стоящих выше СВОЕЙ скользящей средней.

    Средняя обязана считаться ровно по w точкам, заканчивающимся на самой точке.
    Формула с накопительным окном (среднее по всему прошлому до точки) даёт
    другие числа на тех же данных, но выглядит правдоподобно и молча проходит
    и ворота на утечку, и все прочие тесты — здесь она ловится.
    """
    rng = np.random.default_rng(7)
    past = np.cumsum(rng.normal(0.0, 1.0, 400)) + 100.0
    f = row_features(past, dt.date(2024, 3, 5), "TJS", 1, {}, None)

    for w in (20, 60):
        hits = []
        for k in range(w):
            j = len(past) - w + k
            sma = float(past[j - w + 1 : j + 1].mean())
            hits.append(bool(past[j] > sma))
        assert f[f"share_above_sma_{w}"] == pytest.approx(float(np.mean(hits))), w

    # накопительное окно — то, чем это НЕ является: даёт другой ответ
    cumulative = float(
        np.mean([past[j] > past[: j + 1].mean() for j in range(len(past) - 60, len(past))])
    )
    assert f["share_above_sma_60"] != pytest.approx(cumulative)


def test_warmup_covers_the_longest_feature_window() -> None:
    """Прогрев короче самого длинного окна — это молчаливый ноль, а не ошибка.

    При WARMUP = 200 признак `ret_250` на первых строках отдавал бы ровно 0.0:
    не «доходность близка к нулю», а «данных не хватило». Модель училась бы
    на константе, ничего при этом не падало бы.
    """
    longest = max(RETURN_LAGS + RANGE_WINDOWS + VOL_WINDOWS)
    assert WARMUP > longest, (WARMUP, longest)

    past = np.arange(WARMUP, dtype=float) + 100.0  # строго растущий ряд
    f = row_features(past, dt.date(2024, 3, 5), "TJS", 1, {}, None)

    assert f[f"ret_{max(RETURN_LAGS)}"] != 0.0
    assert f["share_above_sma_60"] == 1.0  # не заглушка 0.5
    assert f["range_width_90"] > 0.0
