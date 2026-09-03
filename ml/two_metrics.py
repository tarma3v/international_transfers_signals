"""Две метрики и две модели.

Модель A оптимизирует метрику кейса: «сегодня курс не станет ниже за h публикаций».
Модель B оптимизирует предлагаемую нами метрику клиента: «сколько дополнительных
сомони получит семья по сравнению с переводом в день зарплаты».

Вторая метрика отличается тем, что считается ТОЛЬКО в моменты, когда клиент
действительно принимает решение, и сравнивается с тем, что он сделал бы без нас.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from ml.data import Series

PAYDAYS: tuple[int, ...] = (5, 20)


def payday_anchors(dates: list[dt.date]) -> list[int]:
    """Индексы публикаций, ближайших к типовым датам выплат (5-е и 20-е)."""
    out: list[int] = []
    for i, d in enumerate(dates):
        if d.day in PAYDAYS:
            out.append(i)
        elif i > 0 and d.day in (6, 7, 21, 22) and dates[i - 1].day not in PAYDAYS:
            # выплата попала на выходной — берём первую публикацию после неё
            if not out or out[-1] < i - 4:
                out.append(i)
    return out


def target_case(values: np.ndarray, i: int, h: int) -> float | None:
    """Метрика кейса: курс не станет ниже за h публикаций."""
    if i + h >= len(values):
        return None
    return 1.0 if values[i] <= values[i + 1 : i + h + 1].min() else 0.0


def target_client(values: np.ndarray, i: int, remaining: int) -> float | None:
    """Метрика клиента: насколько сегодня лучше ЛУЧШЕГО из оставшихся дней окна.

    Положительное значение = переводить сейчас, дальше будет только хуже.
    Отрицательное = стоит подождать, впереди есть день лучше.
    Это формулировка задачи оптимальной остановки, а не предсказания курса.
    """
    if remaining <= 0 or i + remaining >= len(values):
        return None
    best_ahead = float(values[i + 1 : i + remaining + 1].min())
    return -(float(values[i]) - best_ahead) / float(values[i]) * 10000.0


def build_windows(
    series: dict[str, Series], corridors: tuple[str, ...], window: int, warmup: int
) -> dict[tuple[str, int], tuple[int, int]]:
    """Для каждой строки — к какому зарплатному окну она относится.

    Возвращает {(коридор, индекс): (индекс дня зарплаты, сколько публикаций до конца окна)}.
    """
    out: dict[tuple[str, int], tuple[int, int]] = {}
    for c in corridors:
        dates = list(series[c].dates)
        n = len(dates)
        for p in payday_anchors(dates):
            if p < warmup or p + window >= n:
                continue
            for k in range(window + 1):
                out[(c, p + k)] = (p, window - k)
    return out


@dataclass(frozen=True)
class PolicyRun:
    """Исход политики по каждому окну — единственный источник истины для отчётов.

    Возвращается вектор, а не среднее: агрегировать (среднее, ДИ с кластеризацией
    по датам, разложение по дню срабатывания) должен вызывающий. Раньше отчёт
    пересчитывал выгоды вторым, параллельным кодом — два источника истины для
    главного числа.
    """

    gains: np.ndarray  # бп против дня зарплаты, по одному на окно
    day_used: np.ndarray  # на какой день окна пришёлся перевод (0 = день зарплаты)
    fired: np.ndarray  # сработал ли сигнал в этом окне (bool)
    anchor: np.ndarray  # индекс дня зарплаты каждого окна

    def __len__(self) -> int:
        return len(self.gains)

    @property
    def mean(self) -> float:
        return float(self.gains.mean()) if len(self.gains) else float("nan")

    @property
    def fire_rate(self) -> float:
        return float(self.fired.mean()) if len(self.fired) else 0.0


def evaluate_policy(
    values: np.ndarray,
    windows: list[tuple[int, int]],
    fires: dict[int, bool],
) -> PolicyRun:
    """Симуляция политики: переводим в первый день окна, где сигнал сработал.

    Если не сработал ни разу — переводим в конце окна (клиент всё равно переведёт).
    """
    gains: list[float] = []
    day_used: list[int] = []
    fired: list[bool] = []
    anchor: list[int] = []
    for p, w in windows:
        chosen = next((p + k for k in range(w + 1) if fires.get(p + k, False)), None)
        fired.append(chosen is not None)
        if chosen is None:
            chosen = p + w
        day_used.append(chosen - p)
        anchor.append(p)
        gains.append(-(float(values[chosen]) - float(values[p])) / float(values[p]) * 10000.0)
    return PolicyRun(np.array(gains), np.array(day_used, dtype=int),
                     np.array(fired, dtype=bool), np.array(anchor, dtype=int))


def oracle_gain(values: np.ndarray, windows: list[tuple[int, int]]) -> float:
    """Потолок: перевод в лучший день окна. Требует знания будущего, недостижим."""
    g = []
    for p, w in windows:
        best = int(np.argmin(values[p : p + w + 1])) + p
        g.append(-(float(values[best]) - float(values[p])) / float(values[p]) * 10000.0)
    return float(np.mean(g)) if g else float("nan")
