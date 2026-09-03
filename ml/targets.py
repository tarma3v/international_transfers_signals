"""Целевые переменные — обе метрики заказчика, все пять горизонтов.

Цели ЛЕГИТИМНО смотрят в будущее: это то, что мы предсказываем. Запрет на
заглядывание относится к признакам. Разделение обеспечивается тем, что цели
живут в отдельном модуле и не участвуют в построении матрицы признаков.
"""
from __future__ import annotations

import numpy as np

from ml.data import Series

HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)


def target_now_favourable(values: np.ndarray, i: int, h: int) -> float | None:
    """«Сейчас выгодно»: курс не станет ниже в ближайшие h публикаций.

    Определение заказчика: v[i] <= min(v[i+1 : i+h+1]). Только вперёд.
    """
    if i + h >= len(values):
        return None
    return 1.0 if values[i] <= values[i + 1 : i + h + 1].min() else 0.0


def target_window_closing(values: np.ndarray, i: int, h: int) -> float | None:
    """«Окно закрывается»: через h публикаций курс окажется выше сегодняшнего."""
    if i + h >= len(values):
        return None
    return 1.0 if values[i + h] > values[i] else 0.0


def benefit_bps(values: np.ndarray, i: int, h: int) -> float | None:
    """«Выгода момента» в бп для отправителя — против СИММЕТРИЧНОГО окна +-h.

    Симметричность — требование заказчика: момент сравнивается с окружением,
    а не только с будущим. Знак инвертирован (рост курса ЦБ = хуже клиенту).
    """
    if i + h >= len(values) or i - h < 0:
        return None
    ref = float(values[i - h : i + h + 1].mean())
    if ref <= 0:
        return None
    return -(float(values[i]) - ref) / ref * 10000.0


def build_targets(
    series: dict[str, Series], index: list[tuple[str, int, object]]
) -> dict[str, np.ndarray]:
    """Все цели для строк матрицы признаков. NaN там, где горизонт не помещается."""
    out: dict[str, np.ndarray] = {}
    for h in HORIZONS:
        for name, fn in (
            ("fav", target_now_favourable),
            ("close", target_window_closing),
            ("benefit", benefit_bps),
        ):
            col = np.full(len(index), np.nan)
            for r, (corridor, i, _) in enumerate(index):
                val = fn(series[corridor].values, i, h)
                if val is not None:
                    col[r] = val
            out[f"{name}_h{h}"] = col
    return out


def benefit_forward_only(values: np.ndarray, i: int, h: int) -> float | None:
    """Достижимая половина выгоды: сравнение ТОЛЬКО с будущим окном.

    Метрика заказчика «выгода момента» симметрична (+-h) и потому наполовину
    описывает уже случившееся падение. Правило, срабатывающее ПОСЛЕ падения,
    набирает по ней много, ничего не предсказав. Клиент может забрать только
    эту, форвардную половину.
    """
    if i + h >= len(values):
        return None
    ref = float(values[i + 1 : i + h + 1].mean())
    if ref <= 0:
        return None
    return -(float(values[i]) - ref) / ref * 10000.0


def benefit_backward_only(values: np.ndarray, i: int, h: int) -> float | None:
    """Недостижимая половина: сравнение с уже прошедшим окном. Забрать нельзя."""
    if i - h < 0:
        return None
    ref = float(values[i - h : i].mean())
    if ref <= 0:
        return None
    return -(float(values[i]) - ref) / ref * 10000.0
