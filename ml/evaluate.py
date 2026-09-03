"""Метрики сравнения. Правило и модель сопоставимы только при равной частоте."""
from __future__ import annotations

import numpy as np


def rate_per_week(
    n_signals: int, n_corridors: int, dates: np.ndarray, scope: np.ndarray
) -> float:
    """Сигналов на коридор в неделю — единственное определение на весь проект.

    Считается по фактическому календарному охвату тех строк, на которых сигнал
    вообще мог сработать (`scope`), а не по округлённому числу лет: разные
    константы в разных скриптах давали расхождение около 2 % в одной и той же
    строке отчёта. Полоса из ТЗ — 1-2 сигнала на коридор в неделю.
    """
    d = [x for x, ok in zip(dates, scope) if ok]
    if not d or n_corridors <= 0:
        return float("nan")
    weeks = (max(d) - min(d)).days / 7.0
    return n_signals / n_corridors / weeks if weeks > 0 else float("nan")


def lift(
    fired: np.ndarray, y: np.ndarray, scope: np.ndarray | None = None
) -> tuple[float, float, int]:
    """lift = доля попаданий среди сработавших / базовая ставка.

    `scope` — строки, на которых сигнал оценивается (обычно out-of-sample).
    Базовая ставка обязана считаться по ним же: если знаменатель взять по всей
    выборке, а срабатывания — по тесту, lift сдвигается на дрейф базовой ставки
    между периодами. На этих данных это давало завышение около 2,6 %.
    """
    ok = ~np.isnan(y)
    if scope is not None:
        ok = ok & scope.astype(bool)
    fired = fired.astype(bool) & ok
    base = float(y[ok].mean()) if ok.sum() else float("nan")
    if fired.sum() == 0 or not base > 0:
        return float("nan"), base, 0
    return float(y[fired].mean()) / base, base, int(fired.sum())


def mean_benefit(fired: np.ndarray, benefit: np.ndarray) -> float:
    ok = ~np.isnan(benefit)
    f = fired.astype(bool) & ok
    return float(benefit[f].mean()) if f.sum() else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Два имени, назначенные ДО теста. Оба раньше подставлялись по месту, и оба
# разъезжались между отчётами.
# ─────────────────────────────────────────────────────────────────────────────

#: Правило, чья частота срабатывания на периоде разработки задаёт целевую
#: частоту для порогов моделей. Оно должно быть одно на весь проект: раньше
#: run_two_models брал «верх диапазона» (9,4 %), а остальные скрипты —
#: «нижний дециль» (10,2 %), и колонки «частота» и «сигн/нед» в двух отчётах
#: были несопоставимы для одних и тех же моделей.
REFERENCE_RULE = "ТЗ: уровень (нижний дециль)"

#: Компаратор для uplift, назначенный ЗАРАНЕЕ. Максимум по четырём индикаторам,
#: посчитанный на тесте, — не парная оценка, а max-статистика: у неё нет
#: распределения, и повторить её на новых данных нельзя. Берём основной
#: индикатор ТЗ — тот, на котором построена сама постановка кейса.
UPLIFT_COMPARATOR = "ТЗ: моментум (падение 3 дн)"


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
    if not 0.0 < rate < 1.0:
        raise ValueError(f"целевая частота вне (0;1): {rate}")
    if len(s) == 0:
        raise ValueError("нет обучающих оценок — порог считать не из чего")
    return float(np.quantile(s, 1.0 - rate))


def bootstrap_ci(
    x: np.ndarray, dates: np.ndarray | None = None, B: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95 % доверительный интервал среднего.

    С `dates` ресемплируются ДНИ целиком, со всеми коридорами внутри. Это
    обязательно: выгода в пяти коридорах в один день коррелирована на 0,86, и
    построчный i.i.d. бутстрап считает пять зависимых наблюдений за пять
    независимых, занижая интервал примерно вдвое. Без `dates` возвращается
    построчный интервал — он годится только там, где наблюдения независимы.
    """
    good = ~np.isnan(x)
    xs = x[good]
    if len(xs) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if dates is None:
        m = np.sort(rng.choice(xs, size=(B, len(xs)), replace=True).mean(axis=1))
        return float(m[int(0.025 * (B - 1))]), float(m[int(0.975 * (B - 1))])

    ds = np.asarray(dates, dtype=object)[good]
    uniq = sorted(set(ds))
    if len(uniq) < 10:
        return float("nan"), float("nan")
    groups = [xs[ds == d] for d in uniq]
    means = np.empty(B)
    n = len(uniq)
    for b in range(B):
        pick = rng.integers(0, n, size=n)
        means[b] = np.concatenate([groups[k] for k in pick]).mean()
    means = np.sort(means)
    # индексация от B-1: means[int(0.975*B)] при B=2000 — это 97,55-й процентиль
    return float(means[int(0.025 * (B - 1))]), float(means[int(0.975 * (B - 1))])
