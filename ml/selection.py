"""Отбор признаков БЕЗ подглядывания в тест.

Ловушка, в которую легко попасть: отобрать признаки по всей выборке, а потом
показать walk-forward. Тест при этом уже участвовал в решении, какие признаки
брать, и результат завышен.

Здесь отбор идёт только по данным ДО первого тестового года. Внутри этого
периода — своя граница: ранжирование на внутреннем обучении, выбор числа
признаков на внутренней валидации. Тест не участвует ни в одном решении.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from sklearn.metrics import roc_auc_score

from ml.models import make_classifiers

CANDIDATE_K: tuple[int, ...] = (10, 15, 20, 30, 45, 60, 999)


def _purge_gap(horizon: int) -> dt.timedelta:
    """Календарный запас — резервная ветка, когда точные даты не переданы.

    Она заведомо неточна: горизонт задан в ПУБЛИКАЦИЯХ, а запас отсчитывается
    в календарных днях. Фактические максимумы по данным ЦБ — 13 дней на h = 1
    и 17 на h = 3 против запаса 12 и 16, то есть запас бывает МЕНЬШЕ нужного.
    Сейчас это не протекает только потому, что внутренняя стенка приходится на
    участок без длинных разрывов; на других данных протечёт. Поэтому обе
    функции ниже принимают `reach` и при его наличии режут точно.
    """
    return dt.timedelta(days=2 * horizon + 10) if horizon > 0 else dt.timedelta(0)


def _inner_split(
    dates: np.ndarray,
    dev: np.ndarray,
    inner_wall: dt.date,
    horizon: int,
    reach: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Внутренний сплит периода разработки с очисткой в публикациях.

    Та же арифметика, что в `ml/validation.py`: обучающая строка выбрасывается,
    если дата, до которой достаёт её цель, попадает за внутреннюю стенку.
    """
    inner_va = dev & np.array([d > inner_wall for d in dates])
    if reach is None:
        inner_tr = dev & np.array([d <= inner_wall - _purge_gap(horizon) for d in dates])
    else:
        inner_tr = dev & np.array(
            [d <= inner_wall and rr <= inner_wall for d, rr in zip(dates, reach)]
        )
    return inner_tr, inner_va


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    names: list[str],
    first_test_year: int,
    horizon: int = 0,
    inner_valid_months: int = 6,
    model_name: str = "CatBoost",
    reach: np.ndarray | None = None,
) -> tuple[list[int], int, list[tuple[int, float]]]:
    """Возвращает (индексы отобранных, выбранное K, отчёт по K)."""
    wall = dt.date(first_test_year, 1, 1)
    dev = np.array([d < wall for d in dates]) & ~np.isnan(y)
    if dev.sum() < 500:
        raise ValueError("мало данных до первого тестового года")

    dev_dates = [d for d in dates[dev]]
    inner_wall = max(dev_dates) - dt.timedelta(days=30 * inner_valid_months)
    inner_tr, inner_va = _inner_split(dates, dev, inner_wall, horizon, reach)
    if inner_tr.sum() < 300 or inner_va.sum() < 100:
        raise ValueError("не удалось разделить период разработки")

    ranker = make_classifiers()[model_name]
    ranker.fit(X[inner_tr], y[inner_tr])
    imp = ranker.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1]

    report: list[tuple[int, float]] = []
    best_k, best_auc = len(names), -1.0
    for k in CANDIDATE_K:
        cols = order[: min(k, len(names))]
        m = make_classifiers()[model_name]
        m.fit(X[inner_tr][:, cols], y[inner_tr])
        auc = roc_auc_score(y[inner_va], m.predict_proba(X[inner_va][:, cols])[:, 1])
        report.append((min(k, len(names)), float(auc)))
        if auc > best_auc:
            best_auc, best_k = auc, min(k, len(names))
    return list(order[:best_k]), best_k, report


def select_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    first_test_year: int,
    cols: list[int] | None = None,
    horizon: int = 0,
    inner_valid_months: int = 6,
    reach: np.ndarray | None = None,
) -> tuple[str, list[tuple[str, float]]]:
    """Выбор модели ДО теста — по внутренней валидации внутри периода разработки.

    Ловушка того же рода, что и порог из теста: обучить пять моделей, посмотреть
    их результат на тесте и назвать «аплифтом» лучшую. Тест тогда участвует в
    решении, какую модель показывать, и цифра завышена тем сильнее, чем больше
    моделей перебрано. Здесь победитель определяется до теста.
    """
    wall = dt.date(first_test_year, 1, 1)
    dev = np.array([d < wall for d in dates]) & ~np.isnan(y)
    if dev.sum() < 500:
        raise ValueError("мало данных до первого тестового года")
    dev_dates = [d for d in dates[dev]]
    inner_wall = max(dev_dates) - dt.timedelta(days=30 * inner_valid_months)
    inner_tr, inner_va = _inner_split(dates, dev, inner_wall, horizon, reach)
    if inner_tr.sum() < 300 or inner_va.sum() < 100:
        raise ValueError("не удалось разделить период разработки")
    use = slice(None) if cols is None else cols

    report: list[tuple[str, float]] = []
    for nm, mdl in make_classifiers().items():
        mdl.fit(X[inner_tr][:, use], y[inner_tr])
        auc = float(roc_auc_score(y[inner_va], mdl.predict_proba(X[inner_va][:, use])[:, 1]))
        report.append((nm, auc))
    report.sort(key=lambda r: -r[1])
    return report[0][0], report
