"""Скептическая проверка найденного. Запускается ПОСЛЕ run_experiment.py.

Четыре испытания, каждое способно отменить результат:
 1. Разложение выгоды на форвардную (достижимую) и обратную (недостижимую)
 2. Нулевая полоса блочным бутстрапом — не шум ли lift
 3. Трейлинг-база вместо глобальной — знаменатель, честный в момент решения
 4. Устойчивость по годам — не 2022 ли делает весь результат
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import REFERENCE_RULE, bootstrap_ci, lift, reference_rate, train_cutoff
from ml.features import build_matrix
from ml.models import make_classifiers
from ml.targets import (
    benefit_backward_only,
    benefit_bps,
    benefit_forward_only,
    build_targets,
)
from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds

H = 5
TARGET = "fav"
rng = np.random.default_rng(17)


def oos_scores(X, y, dates, h, rate: float, reach, index, series):
    """Частота передаётся параметром, а не читается из модуля.

    Модульная глобальная переменная здесь опасна: при импорте функция взяла бы
    заглушку вместо реальной частоты, и любая перестановка строк в main() меняла
    бы результат молча, без исключения.
    """
    folds = walk_forward_folds(dates, 2021, h, reach=reach)
    oos = np.zeros(len(y), dtype=bool)
    sc = {m: np.full(len(y), np.nan) for m in make_classifiers()}
    fr = {m: np.zeros(len(y), bool) for m in make_classifiers()}
    for tr_i, te_i, _ in folds:
        assert_no_overlap(dates, tr_i, te_i, h, index=index, series=series)
        tr = tr_i[~np.isnan(y[tr_i])]
        te = te_i[~np.isnan(y[te_i])]
        if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
            continue
        oos[te] = True
        for m, mdl in make_classifiers().items():
            mdl.fit(X[tr], y[tr])
            sc[m][te] = mdl.predict_proba(X[te])[:, 1]
            # рабочая точка — из обучения фолда, а не из распределения оценок теста
            fr[m][te] = sc[m][te] >= train_cutoff(mdl.predict_proba(X[tr])[:, 1], rate)
    return oos, sc, fr


def main() -> None:
    s = load()
    X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
    X = np.column_stack([X, np.array([CORRIDORS.index(c) for c, _, _ in index], float)])
    dates = np.array([d for _, _, d in index], dtype=object)
    Y = build_targets(s, index)
    y = Y[f"{TARGET}_h{H}"]
    rate = reference_rate(
        BASELINES[REFERENCE_RULE](X[:, :-1], names), dates, 2021)
    reach = target_reach_dates(index, s, H)

    fwd = np.full(len(index), np.nan)
    bwd = np.full(len(index), np.nan)
    sym = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        a = benefit_forward_only(s[c].values, i, H)
        b = benefit_backward_only(s[c].values, i, H)
        m = benefit_bps(s[c].values, i, H)
        if m is not None:
            sym[r] = m
        if a is not None:
            fwd[r] = a
        if b is not None:
            bwd[r] = b

    oos, sc, fr = oos_scores(X, y, dates, H, rate, reach, index, s)
    base = float(np.nanmean(y[oos]))

    print("=" * 104)
    print(f"ИСПЫТАНИЕ 1. РАЗЛОЖЕНИЕ ВЫГОДЫ  (цель «сейчас выгодно», h={H})")
    print("Клиент может забрать только форвардную половину. Обратная — уже случилась.")
    print("=" * 104)
    print("Колонка «выгода ±h» — это `benefit_bps`, ровно та метрика, которую печатают")
    print("run_experiment и summarize. Полусумма форварда и обратной ей НЕ равна:")
    print("половинки нормированы на разные базы, и на тренде это меняет даже знак.")
    print(f"{'правило / модель':<34}{'выгода ±h':>12}{'форвард':>11}{'95% ДИ форв.':>19}{'обратная':>12}")
    fired_map: dict[str, np.ndarray] = {}
    for bn, bf in BASELINES.items():
        fired_map[bn] = bf(X[:, :-1], names).astype(bool) & oos
    for m in sc:
        fired_map["* " + m] = fr[m] & oos
    for nm, f in fired_map.items():
        tot = np.nanmean(np.where(f, sym, np.nan))
        a = np.nanmean(np.where(f, fwd, np.nan))
        b = np.nanmean(np.where(f, bwd, np.nan))
        sel = f & ~np.isnan(fwd)
        lo, hi = bootstrap_ci(fwd[sel], dates=dates[sel])
        print(f"{nm:<34}{tot:>+11.0f}бп{a:>+10.0f}бп{f'[{lo:+.0f}; {hi:+.0f}]':>19}{b:>+11.0f}бп")

    print("\n" + "=" * 104)
    print("ИСПЫТАНИЕ 2. НУЛЕВАЯ ПОЛОСА (блочный бутстрап, случайные наборы дней той же численности)")
    print("=" * 104)
    idx_ok = np.where(oos & ~np.isnan(y))[0]
    by_day: dict[object, list[int]] = {}
    for r in idx_ok:
        by_day.setdefault(dates[r], []).append(int(r))
    day_blocks = [np.array(v, dtype=int) for v in by_day.values()]

    def null_band(n_fire: int) -> tuple[float, float, float]:
        """Полоса нуля для НАБОРА ТОЙ ЖЕ ЧИСЛЕННОСТИ, что у самой модели.

        Блок — это ДЕНЬ целиком, все пять коридоров сразу. Блок из подряд
        идущих строк матрицы был бы не тем: строки идут коридор-мажорно, и
        такой «блок» — это три публикации ОДНОГО коридора подряд. Он не
        учитывает корреляцию между коридорами в один день (0,86 по выгоде,
        0,70 по бинарной цели), и полоса нуля выходит примерно вдвое уже,
        чем на самом деле — а по ней выносится вердикт «значимо».

        Численность фиксирована по числу сработавших строк: с честным порогом
        модели срабатывают с разной частотой, а ширина полосы зависит от неё.
        """
        if n_fire < 10:
            # без срабатываний полоса не определена: индексация пустого массива
            # даёт IndexError, а NaN честно говорит «оценки нет»
            return float("nan"), float("nan"), float("nan")
        null = []
        for _ in range(1000):
            picked: list[int] = []
            while len(picked) < n_fire:
                picked += list(day_blocks[int(rng.integers(0, len(day_blocks)))])
            null.append(y[np.array(picked[:n_fire], dtype=int)].mean() / base)
        null = np.sort(null)
        return float(null[500]), float(null[950]), float(null[990])

    print(f"{'модель':<34}{'частота':>9}{'lift':>8}{'p95':>7}{'p99':>7}{'вердикт':>16}")
    for m in sc:
        f = fired_map["* " + m]
        n_fire = int((f & ~np.isnan(y)).sum())
        lf, _, _ = lift(f, y, scope=oos)
        _p50, p95, p99 = null_band(n_fire)
        # NaN-полоса означает «оценки нет», а не «внутри шума»: любое сравнение
        # с NaN ложно, поэтому без этой ветки редкая модель получила бы
        # успокаивающий вердикт там, где полосу вообще не удалось построить.
        if np.isnan(p95) or np.isnan(p99):
            v = "оценки нет"
        else:
            v = "ВЫШЕ p99" if lf > p99 else ("выше p95" if lf > p95 else "внутри шума")
        print(f"{m:<34}{n_fire/len(idx_ok):>8.0%}{lf:>8.2f}{p95:>7.2f}{p99:>7.2f}{v:>16}")

    print("\n" + "=" * 104)
    print("ИСПЫТАНИЕ 3. ТРЕЙЛИНГ-БАЗА вместо глобальной (знаменатель, известный в момент решения)")
    print("=" * 104)
    print("  База считается только по строкам, чья цель УЖЕ РАЗРЕШИЛАСЬ к дате решения:")
    print(f"  условие reach[q] < d0, а не dates[q] < d0. Цель на горизонте H={H} публикаций")
    print("  становится известна лишь через H публикаций после дня решения, поэтому окно по")
    print("  датам решения включало бы в знаменатель ещё не наступившие исходы.")
    print(f"{'модель / правило':<34}{'lift глоб.':>12}{'lift трейл.':>13}")
    order = np.argsort([d for d in dates])
    for nm, f in fired_map.items():
        lf, _, _ = lift(f, y, scope=oos)
        tb = []
        for r in np.where(f & ~np.isnan(y))[0]:
            d0 = dates[r]
            prev = [
                y[q]
                for q in order
                if reach[q] < d0
                and dates[q] >= d0 - dt.timedelta(days=90)
                and not np.isnan(y[q])
            ]
            if prev:
                tb.append(np.mean(prev))
        tl = float(np.nanmean(np.where(f & ~np.isnan(y), y, np.nan))) / np.mean(tb) if tb else np.nan
        print(f"{nm:<34}{lf:>12.2f}{tl:>13.2f}")

    print("\n" + "=" * 104)
    print("ИСПЫТАНИЕ 4. УСТОЙЧИВОСТЬ ПО ГОДАМ (не 2022 ли делает весь результат)")
    print("=" * 104)
    yrs = sorted({d.year for d in dates[oos]})
    print(f"{'модель':<34}" + "".join(f"{y_:>9}" for y_ in yrs))
    for m, s_ in sc.items():
        f = fired_map["* " + m]
        row = f"{m:<34}"
        for y_ in yrs:
            msk = np.array([d.year == y_ for d in dates]) & f & ~np.isnan(y)
            allm = np.array([d.year == y_ for d in dates]) & oos & ~np.isnan(y)
            row += f"{(y[msk].mean()/y[allm].mean()):>9.2f}" if msk.sum() > 20 else f"{'—':>9}"
        print(row)
    print("\n  Значения — lift внутри года. Устойчивость важнее среднего.")


if __name__ == "__main__":
    main()
