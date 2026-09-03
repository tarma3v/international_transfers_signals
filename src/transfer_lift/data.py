"""Data loading utilities for normalized CBR FX rates."""
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
    """One currency corridor: publication dates and RUB per 1 foreign currency unit."""

    code: str
    dates: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError(f"{self.code}: dates and values lengths differ")
        if len(self.dates) == 0:
            raise ValueError(f"{self.code}: empty series")
        if not all(a < b for a, b in zip(self.dates[:-1], self.dates[1:])):
            raise ValueError(f"{self.code}: dates must be strictly increasing")
        if not np.all(np.isfinite(self.values)) or np.any(self.values <= 0):
            raise ValueError(f"{self.code}: values must be positive finite numbers")

    def __len__(self) -> int:
        return len(self.values)

    def truncate(self, n: int) -> "Series":
        """First n observations only. Used to build strictly as-of-T series slices."""
        return Series(self.code, self.dates[:n].copy(), self.values[:n].copy())


def default_data_path() -> Path:
    """Return the sibling-project CBR JSON path from common working directories."""
    candidates = [
        Path("international_transfers_signals/data/cbr_rates.json"),
        Path("../international_transfers_signals/data/cbr_rates.json"),
        Path(__file__).resolve().parents[3] / "international_transfers_signals/data/cbr_rates.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_rates(path: str | Path | None = None) -> dict[str, Series]:
    """Load normalized JSON rates produced by the existing sibling data loader."""
    data_path = Path(path) if path is not None else default_data_path()
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    series: dict[str, Series] = {}
    for code, rows in raw.items():
        sorted_rows = sorted(rows, key=lambda row: row[0])
        dates = np.array([dt.date.fromisoformat(day) for day, _ in sorted_rows], dtype=object)
        values = np.array([float(value) for _, value in sorted_rows], dtype=np.float64)
        series[code] = Series(code=code, dates=dates, values=values)
    missing = set(CORRIDORS + REFERENCE) - set(series)
    if missing:
        raise ValueError(f"missing required series: {sorted(missing)}")
    return series
