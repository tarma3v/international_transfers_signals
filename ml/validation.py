"""Walk-forward с очисткой (purge). Обучение только на прошлом.

Две защиты:
1. Разделение строго по КАЛЕНДАРНОЙ ДАТЕ, а не по строкам: коридоры сильно
   коррелированы, и случайное разбиение перемешало бы один и тот же день.
2. Очистка: строка обучения, чья цель считается по h дней вперёд, выбрасывается,
   если её окно достаёт до теста. Без этого модель видит тестовый период
   через собственную целевую переменную.
"""
from __future__ import annotations

import datetime as dt

import numpy as np


def walk_forward_folds(
    dates: np.ndarray, first_test_year: int, horizon: int, embargo_days: int = 0
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Расширяющееся окно: тест — календарный год, обучение — всё до него."""
    years = sorted({d.year for d in dates})
    folds = []
    for y in years:
        if y < first_test_year:
            continue
        test_start = dt.date(y, 1, 1)
        test_end = dt.date(y, 12, 31)
        is_test = np.array([test_start <= d <= test_end for d in dates])
        if is_test.sum() < 50:
            continue
        # очистка: цель обучающей строки не должна доставать до теста
        cutoff = test_start - dt.timedelta(days=horizon + embargo_days + 4)
        is_train = np.array([d <= cutoff for d in dates])
        if is_train.sum() < 500:
            continue
        folds.append((np.where(is_train)[0], np.where(is_test)[0], y))
    return folds


def assert_no_overlap(
    dates: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, horizon: int
) -> None:
    """Жёсткая проверка очистки: падает, если обучение достаёт до теста."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("пустой фолд")
    last_train = max(dates[i] for i in train_idx)
    first_test = min(dates[i] for i in test_idx)
    reach = last_train + dt.timedelta(days=horizon)
    if reach >= first_test:
        raise AssertionError(
            f"очистка нарушена: обучение достаёт до {reach}, тест с {first_test}"
        )
