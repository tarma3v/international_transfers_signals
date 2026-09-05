"""Тесты расчёта устаревания сигнала.

Каждый тест здесь закрывает ошибку, которая уже была допущена в первой редакции
`run_signal_staleness.py` и попала в документы. Поэтому они не про «код
работает», а про конкретные способы получить слишком уверенный ответ.
"""
from __future__ import annotations

import numpy as np
import pytest

import run_signal_staleness as st
from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.features import build_matrix
from ml.targets import benefit_forward_only


@pytest.fixture(scope="module")
def matrix():
    series = load()
    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    return series, X, names, index


def test_index_position_is_not_row_number(matrix):
    """i из index — позиция в ряду коридора, а не номер строки матрицы.

    Первая редакция шагала по номеру строки и читала курс на длину прогрева
    раньше. Ошибка не падала, а давала правдоподобное неверное число.
    """
    _series, _X, _names, index = matrix
    first_row_per_corridor = {}
    for row, (c, i, _d) in enumerate(index):
        first_row_per_corridor.setdefault(c, (row, i))
    # у первого коридора номер строки и i совпадают только случайно,
    # у остальных — расходятся на длину предыдущих коридоров
    offsets = {c: row - i for c, (row, i) in first_row_per_corridor.items()}
    assert len(set(offsets.values())) > 1, (
        "если бы смещение было одинаковым, подмена i номером строки была бы "
        "безобидной — тест потерял бы смысл"
    )


def test_cohort_is_identical_across_delays(matrix):
    """Все строки таблицы считаются по одному и тому же набору сигналов.

    Отсечка «горизонт не помещается» своя у каждой задержки. Если её применять
    построчно, k=3 получает лишние наблюдения, которых нет у k=5, и средние
    становятся несравнимыми: именно эти пять лишних строк в первой редакции
    поднимали остаток выше порога запуска.
    """
    series, X, names, index = matrix
    fires = BASELINES[st.RULE](X, names).astype(bool)
    kmax = max(st.DELAYS)
    cohort = [
        (c, i) for row, (c, i, d) in enumerate(index)
        if fires[row] and d.year >= st.FIRST_TEST
        and i + kmax + st.H < len(series[c].values)
    ]
    counts = set()
    for k in st.DELAYS:
        n = sum(
            1 for c, i in cohort
            if benefit_forward_only(series[c].values, i + k, st.H) is not None
        )
        counts.add(n)
    assert len(counts) == 1, f"когорта разъезжается по задержкам: {counts}"


def test_block_bootstrap_is_wider_than_independent_days():
    """Блочный бутстрап шире построчного там, где соседние дни зависимы.

    benefit_forward_only(i) смотрит на v[i+1..i+H], поэтому соседние публикации
    делят H−1 будущих курсов. Ресемплирование отдельных дат эту зависимость
    игнорирует и занижает интервал.
    """
    rng = np.random.default_rng(7)
    n = 400
    # сильно автокоррелированный ряд: соседние даты почти одинаковы
    walk = np.cumsum(rng.normal(0, 1, n))
    sig_sum = walk.copy()
    sig_n = np.ones(n)
    bg_sum = np.zeros(n)
    bg_n = np.ones(n)
    lo1, hi1 = st.block_bootstrap_diff(sig_sum, sig_n, bg_sum, bg_n, 1, 0.05)
    lo5, hi5 = st.block_bootstrap_diff(sig_sum, sig_n, bg_sum, bg_n, st.H, 0.05)
    assert (hi5 - lo5) > (hi1 - lo1), (
        f"блочный интервал {hi5 - lo5:.2f} не шире построчного {hi1 - lo1:.2f}"
    )


def test_bootstrap_returns_nan_on_tiny_sample():
    """Мало наблюдений — не молчаливый интервал, а явный nan.

    Иначе пустой результат читается как «сигнал с фоном не сравнялся».
    """
    small = np.ones(3)
    lo, hi = st.block_bootstrap_diff(small, small, small, small, st.H, 0.05)
    assert np.isnan(lo) and np.isnan(hi)


def test_alpha_is_corrected_for_number_of_delays():
    """Пять маргинальных 95 % интервалов не дают 95 % одновременного покрытия."""
    assert st.ALPHA == pytest.approx(0.05 / len(st.DELAYS))
    assert st.ALPHA < 0.05
