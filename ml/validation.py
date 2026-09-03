"""Walk-forward с очисткой (purge). Обучение только на прошлом.

Две защиты:
1. Разделение строго по КАЛЕНДАРНОЙ ДАТЕ, а не по строкам: коридоры сильно
   коррелированы, и случайное разбиение перемешало бы один и тот же день.
2. Очистка: строка обучения, чья цель считается по h ПУБЛИКАЦИЙ вперёд,
   выбрасывается, если её окно достаёт до теста. Без этого модель видит
   тестовый период через собственную целевую переменную.

Горизонт измеряется в публикациях, а не в календарных днях, и очистка обязана
измеряться в тех же единицах. Курсы ЦБ выходят по рабочим дням, поэтому 20
публикаций занимают до 40 календарных дней: запас, отсчитанный в днях, оставляет
в обучении строки, чья цель уже лежит внутри теста. Точные даты достижения даёт
`target_reach_dates`, и обе функции ниже принимают их параметром `reach`.
"""
from __future__ import annotations

import datetime as dt

import numpy as np


def target_reach_dates(index: list, series: dict, horizon: int) -> np.ndarray:
    """Календарная дата, до которой достаёт целевая переменная каждой строки.

    Считается по фактическому ряду публикаций коридора, а не по календарю.
    """
    out = np.empty(len(index), dtype=object)
    for r, (c, i, _) in enumerate(index):
        ds = series[c].dates
        out[r] = ds[min(i + horizon, len(ds) - 1)]
    return out


def walk_forward_folds(
    dates: np.ndarray,
    first_test_year: int,
    horizon: int,
    reach: np.ndarray | None = None,
    embargo_days: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Расширяющееся окно: тест — календарный год, обучение — всё до него.

    `reach` — даты из `target_reach_dates`. Без них берётся заведомо
    консервативная календарная оценка (2h + 10 дней), потому что h публикаций
    никогда не укладываются в h дней.
    """
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
        emb = dt.timedelta(days=embargo_days)
        if reach is None:
            cutoff = test_start - dt.timedelta(days=2 * horizon + 10 + embargo_days)
            is_train = np.array([d <= cutoff for d in dates])
        else:
            is_train = np.array(
                [d < test_start and rr + emb < test_start for d, rr in zip(dates, reach)]
            )
        if is_train.sum() < 500:
            continue
        folds.append((np.where(is_train)[0], np.where(is_test)[0], y))
    return folds


def _reach_recomputed(index: list, series: dict, horizon: int, rows: np.ndarray) -> np.ndarray:
    """Та же величина, что и `target_reach_dates`, но посчитанная от ДАТЫ строки.

    Сторож не имеет права пользоваться массивом, который он проверяет, —
    иначе он повторяет вычисление, а не проверяет его. Поэтому здесь позиция
    публикации в ряду ищется заново, поиском по дате, а не берётся из
    `index[r][1]`: если бы индекс строки разъехался с рядом, сравнение двух
    копий одной и той же ошибки промолчало бы.
    """
    cache = {c: np.array(s.dates, dtype=object) for c, s in series.items()}
    out = np.empty(len(rows), dtype=object)
    for k, r in enumerate(rows):
        c, _i, day = index[r]
        ds = cache[c]
        pos = int(np.searchsorted(ds, day, side="left"))
        out[k] = ds[min(pos + horizon, len(ds) - 1)]
    return out


def assert_no_overlap(
    dates: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int,
    index: list | None = None,
    series: dict | None = None,
) -> None:
    """Жёсткая проверка очистки: падает, если обучение достаёт до теста.

    С `index` и `series` сторож пересчитывает достижимые даты сам, независимым
    кодом (`_reach_recomputed`), и потому способен упасть. Передавать сюда тот
    же массив `reach`, которым сделана очистка, было бы тавтологией: обе
    стороны сравнения содержали бы одну и ту же ошибку.

    Без `index`/`series` остаётся календарная оценка. Она заведомо слабее —
    h публикаций не укладываются в h календарных дней, — поэтому такой вызов
    считается неполной проверкой и печатать «утечки нет» по нему нельзя.
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("пустой фолд")
    first_test = min(dates[i] for i in test_idx)
    if index is None or series is None:
        worst = max(dates[i] for i in train_idx) + dt.timedelta(days=horizon)
    else:
        worst = max(_reach_recomputed(index, series, horizon, np.asarray(train_idx)))
    if worst >= first_test:
        raise AssertionError(
            f"очистка нарушена: цель обучения достаёт до {worst}, тест с {first_test}"
        )
