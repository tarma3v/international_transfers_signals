"""Метрики сравнения. Правило и модель сопоставимы только при равной частоте."""
from __future__ import annotations

import numpy as np


def rate_per_week(n_signals: int, n_rows: int, n_corridors: int, years: float) -> float:
    """Сигналов на коридор в неделю — для проверки полосы 1-2/нед."""
    del n_rows
    return n_signals / n_corridors / (years * 52.0)


def lift(fired: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """lift = доля попаданий среди сработавших / базовая ставка."""
    ok = ~np.isnan(y)
    fired = fired.astype(bool) & ok
    base = float(y[ok].mean())
    if fired.sum() == 0 or base <= 0:
        return float("nan"), base, 0
    return float(y[fired].mean()) / base, base, int(fired.sum())


def mean_benefit(fired: np.ndarray, benefit: np.ndarray) -> float:
    ok = ~np.isnan(benefit)
    f = fired.astype(bool) & ok
    return float(benefit[f].mean()) if f.sum() else float("nan")


def reference_rate(
    fired_rule: np.ndarray,
    dates: np.ndarray,
    first_test_year: int,
) -> float:
    """Целевая частота срабатываний — доля дней, в которые срабатывает правило
    сравнения, посчитанная на периоде РАЗРАБОТКИ.

    Определение одно на весь проект: круглая константа, вписанная в каждый скрипт
    руками, разъезжается между отчётами, а доля, посчитанная на тесте, была бы
    ещё одним заглядыванием вперёд.
    """
    import datetime as _dt

    pre = np.array([d < _dt.date(first_test_year, 1, 1) for d in dates])
    return float(fired_rule.astype(bool)[pre].mean())


def train_cutoff(score_train: np.ndarray, rate: float) -> float:
    """Порог срабатывания, посчитанный ТОЛЬКО по обучающим оценкам.

    Рабочая точка — такое же решение, как выбор признаков, и брать её из теста
    нельзя. Квантиль по тестовым оценкам даёт ровно ту частоту, что у правила
    сравнения, и таблица выглядит аккуратно — но такой порог знает распределение
    оценок на тестовом периоде, и результат оказывается завышен. Ворота на утечку
    этого не ловят по конструкции: они проверяют матрицу признаков, а не то, как
    выбрана рабочая точка.

    Фактическая частота на тесте получится другой, чем целевая, и это нормально:
    сравнение честно ровно тогда, когда частоту не подгоняли под тест.
    """
    s = score_train[~np.isnan(score_train)]
    if len(s) == 0 or rate <= 0 or rate >= 1:
        return float("inf")
    return float(np.quantile(s, 1.0 - rate))


def bootstrap_ci(x: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = np.sort(rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1))
    return float(m[int(0.025 * B)]), float(m[int(0.975 * B)])
