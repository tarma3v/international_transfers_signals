"""Загрузка курсов ЦБ. Ряд публикаций, без календарной сетки и без forward-fill."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORRIDORS: tuple[str, ...] = ("TJS", "UZS", "KGS", "AMD", "KZT")
REFERENCE: tuple[str, ...] = ("USD", "CNY")


@dataclass(frozen=True)
class Series:
    """Один коридор: даты публикаций и курс RUB за 1 единицу валюты.

    Значения уже нормированы построчно (value/nominal) на этапе выгрузки:
    номинал ЦБ менялся в середине истории (TJS 1->10, UZS 1000->10000,
    KGS 10->100, CNY 1->10), и постоянный номинал дал бы разрыв в 10 раз.
    """

    code: str
    dates: np.ndarray  # dtype=object, datetime.date, строго возрастающие
    values: np.ndarray  # float64, RUB за 1 единицу

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError(f"{self.code}: длины дат и значений расходятся")
        if not all(a < b for a, b in zip(self.dates[:-1], self.dates[1:])):
            raise ValueError(f"{self.code}: даты не строго возрастают")
        if not np.all(np.isfinite(self.values)) or np.any(self.values <= 0):
            raise ValueError(f"{self.code}: неположительные или нечисловые курсы")

    def __len__(self) -> int:
        return len(self.values)

    def truncate(self, n: int) -> "Series":
        """Первые n наблюдений. Используется тестом на утечку."""
        return Series(self.code, self.dates[:n].copy(), self.values[:n].copy())


def load(path: str | Path = "data/cbr_rates.json") -> dict[str, Series]:
    raw = json.loads(Path(path).read_text())
    out: dict[str, Series] = {}
    for code, rows in raw.items():
        rows = sorted(rows, key=lambda r: r[0])
        dates = np.array([dt.date.fromisoformat(d) for d, _ in rows], dtype=object)
        values = np.array([float(v) for _, v in rows], dtype=np.float64)
        out[code] = Series(code, dates, values)
    missing = set(CORRIDORS + REFERENCE) - set(out)
    if missing:
        raise ValueError(f"в выгрузке нет коридоров: {sorted(missing)}")
    return out
