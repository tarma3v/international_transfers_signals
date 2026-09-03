"""Базовые индикаторы из постановки задачи — то, с чем сравнивается модель.

Четыре индикатора ТЗ: моментум, уровень, разворот, сезонность. Плюс правило,
которое мы нашли сами (нижний дециль диапазона), и случайный день как нуль.
Все — функции от УЖЕ ПОСЧИТАННЫХ признаков, поэтому наследуют их честность.
"""
from __future__ import annotations

import numpy as np


def _col(X: np.ndarray, names: list[str], name: str) -> np.ndarray:
    return X[:, names.index(name)]


def momentum_down_3(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Индикатор ТЗ «моментум»: курс падает три публикации подряд."""
    return (_col(X, names, "streak_dn") >= 3).astype(float)


def level_low_decile(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Индикатор ТЗ «уровень»: курс в нижних 10 % 90-дневного диапазона."""
    return (_col(X, names, "pct_range_90") <= 10.0).astype(float)


def reversal_from_low(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Индикатор ТЗ «разворот»: минимум окна был недавно, курс пошёл вверх."""
    return (
        (_col(X, names, "bars_since_min_30") <= 3) & (_col(X, names, "streak_up") >= 1)
    ).astype(float)


def seasonality_pre_holiday(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Индикатор ТЗ «сезонность»: две недели до праздника коридора."""
    return _col(X, names, "pre_holiday_14d")


def level_high_decile(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Простое правило: верхние 5 % 90-дневного диапазона.

    Следование тренду: курс на максимуме диапазона чаще продолжает расти, поэтому
    «перевести сейчас» выигрывает у ожидания. Это зеркало индикатора ТЗ «уровень:
    нижний дециль» — и оно обгоняет и его, и все модели. Одна строка кода.
    """
    return (_col(X, names, "pct_range_90") >= 95.0).astype(float)


BASELINES = {
    "ТЗ: моментум (падение 3 дн)": momentum_down_3,
    "ТЗ: уровень (нижний дециль)": level_low_decile,
    "ТЗ: разворот от минимума": reversal_from_low,
    "ТЗ: сезонность (до праздника)": seasonality_pre_holiday,
    "простое правило: верх диапазона": level_high_decile,
}
