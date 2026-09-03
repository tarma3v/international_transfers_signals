"""Две модели под две метрики.

A — оптимизирует метрику кейса. B — оптимизирует метрику клиента.
Обе оцениваются по ОБЕИМ метрикам, чтобы разница была видна, а не заявлена.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import bootstrap_ci, lift, train_cutoff
from ml.features import build_matrix
from ml.leakage import check_detector_works, check_no_lookahead
from ml.models import make_classifiers
from ml.two_metrics import (
    build_windows,
    evaluate_policy,
    oracle_gain,
    payday_anchors,
    target_case,
    target_client,
)
from ml.validation import assert_no_overlap, walk_forward_folds

from catboost import CatBoostClassifier, CatBoostRegressor

# Конфигурации подобраны экспериментально (см. docs/04-dve-metriki-dve-modeli.md).
# Модель A: метрика кейса — монотонная функция уровня, и добавление признаков ей
# только вредит (80 признаков -> lift 1,14; один монотонный -> 1,30).
MODEL_A_FEATURES = ("pct_range_90",)
# Модель B: клиентская метрика; на семи признаках результат лучше, чем на всех.
MODEL_B_FEATURES = (
    "pct_range_90", "pct_range_180", "ret_5", "ret_20", "vol_30", "streak_up", "streak_dn",
)

H = 5              # горизонт метрики кейса
WINDOW = 5         # окно гибкости клиента после зарплаты, публикаций
FIRST_TEST = 2021
SEED = 42


def gate(s) -> None:
    if not check_detector_works(s, dt.date(2023, 6, 30)):
        sys.exit("ОСТАНОВ: проверка не ловит подставную утечку")
    for cut, f in ((dt.date(2023, 6, 30), 3.0), (dt.date(2021, 3, 15), 0.2)):
        ok, leaked, _ = check_no_lookahead(s, cut, f)
        if not ok:
            sys.exit(f"ОСТАНОВ: утечка в признаках: {leaked}")
    print("ВОРОТА: заглядывания в будущее нет\n")


def main() -> None:
    s = load()
    gate(s)
    X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
    X = np.column_stack([X, np.array([CORRIDORS.index(c) for c, _, _ in index], float)])
    names = names + ["corridor_id"]
    dates = np.array([d for _, _, d in index], dtype=object)
    row_of = {(c, i): r for r, (c, i, _) in enumerate(index)}

    # ——— цели ———
    y_case = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        t = target_case(s[c].values, i, H)
        if t is not None:
            y_case[r] = t

    win_map = build_windows(s, CORRIDORS, WINDOW, warmup=200)
    y_client = np.full(len(index), np.nan)
    in_window = np.zeros(len(index), bool)
    for (c, i), (_p, remaining) in win_map.items():
        r = row_of.get((c, i))
        if r is None:
            continue
        in_window[r] = True
        t = target_client(s[c].values, i, remaining)
        if t is not None:
            y_client[r] = t

    print("=" * 104)
    print(f"ДАННЫЕ: строк {len(index)}, признаков {X.shape[1]}")
    print(f"  метрика кейса   — определена на {int((~np.isnan(y_case)).sum())} строках (все дни)")
    print(f"  метрика клиента — определена на {int((~np.isnan(y_client)).sum())} строках "
          f"(только зарплатные окна, {WINDOW} публикаций после выплаты)")
    print("=" * 104 + "\n")

    colsA = [names.index(n) for n in MODEL_A_FEATURES]
    colsB = [names.index(n) for n in MODEL_B_FEATURES]
    # Порог срабатывания фиксируется на ОБУЧЕНИИ каждого фолда.
    # Квантиль по тестовым оценкам — тоже заглядывание вперёд: оно не в признаках,
    # а в выборе рабочей точки, и завышает lift. Поэтому порог считается только
    # по обучающей выборке (см. ml/evaluate.py::train_cutoff).
    fires_A = np.zeros(len(index), bool)
    fires_B = np.zeros(len(index), bool)

    # ——— walk-forward: обучаем обе модели ———
    folds = walk_forward_folds(dates, FIRST_TEST, max(H, WINDOW))
    oos = np.zeros(len(index), bool)
    score_A = np.full(len(index), np.nan)   # вероятность метрики кейса
    score_B = np.full(len(index), np.nan)   # предсказанное преимущество «сегодня vs лучший остаток»

    for tr_i, te_i, _yr in folds:
        assert_no_overlap(dates, tr_i, te_i, max(H, WINDOW))
        oos[te_i] = True

        trA = tr_i[~np.isnan(y_case[tr_i])]
        teA = te_i[~np.isnan(y_case[te_i])]
        if len(trA) > 400 and len(np.unique(y_case[trA])) > 1:
            mA = CatBoostClassifier(iterations=300, depth=2, learning_rate=0.05,
                                    l2_leaf_reg=10.0, monotone_constraints=[1] * len(colsA),
                                    random_seed=SEED, verbose=0, allow_writing_files=False)
            mA.fit(X[trA][:, colsA], y_case[trA])
            score_A[teA] = mA.predict_proba(X[teA][:, colsA])[:, 1]
            thr_train = float(np.quantile(mA.predict_proba(X[trA][:, colsA])[:, 1], 1 - 0.13))
            fires_A[teA] = score_A[teA] >= thr_train

        trB = tr_i[~np.isnan(y_client[tr_i])]
        teB = te_i[~np.isnan(y_client[te_i])]
        if len(trB) > 200:
            mB = CatBoostRegressor(iterations=300, depth=3, learning_rate=0.04,
                                   l2_leaf_reg=10.0, random_seed=SEED, verbose=0,
                                   allow_writing_files=False)
            mB.fit(X[trB][:, colsB], y_client[trB])
            score_B[teB] = mB.predict(X[teB][:, colsB])
            # Та же рабочая точка, что у модели A, и тоже из обучения фолда.
            fires_B[teB] = score_B[teB] >= train_cutoff(mB.predict(X[trB][:, colsB]), 0.13)

    # ——— окна для симуляции политики, только out-of-sample ———
    per_corr_windows: dict[str, list[tuple[int, int]]] = {}
    for c in CORRIDORS:
        dts = list(s[c].dates)
        n = len(dts)
        ws = []
        for p in payday_anchors(dts):
            r = row_of.get((c, p))
            if p < 200 or p + WINDOW >= n or r is None or not oos[r]:
                continue
            ws.append((p, WINDOW))
        per_corr_windows[c] = ws
    total_windows = sum(len(v) for v in per_corr_windows.values())

    def policy_gain(fire_fn) -> tuple[float, float, float, float]:
        """fire_fn(corridor, idx) -> bool. Возвращает (выгода, ДИ низ, ДИ верх, доля окон со срабатыванием)."""
        all_gains: list[float] = []
        fired = 0
        wins = 0
        for c in CORRIDORS:
            v = s[c].values
            fires = {}
            for p, w in per_corr_windows[c]:
                for k in range(w + 1):
                    fires[p + k] = fire_fn(c, p + k)
            g, fr, _day, n = evaluate_policy(v, per_corr_windows[c], fires)
            if n:
                for p, w in per_corr_windows[c]:
                    chosen = next((p + k for k in range(w + 1) if fires.get(p + k)), p + w)
                    all_gains.append(-(float(v[chosen]) - float(v[p])) / float(v[p]) * 10000.0)
                fired += fr * len(per_corr_windows[c])
                wins += len(per_corr_windows[c])
        lo, hi = bootstrap_ci(np.array(all_gains))
        return float(np.mean(all_gains)), lo, hi, fired / max(wins, 1)

    oracle = float(np.mean([oracle_gain(s[c].values, per_corr_windows[c]) for c in CORRIDORS]))

    # ═══ МЕТРИКА КЕЙСА ═══
    print("=" * 104)
    print(f"МЕТРИКА КЕЙСА: «сейчас выгодно», h = {H}. Сравнение при ОДИНАКОВОЙ частоте срабатываний.")
    print("=" * 104)
    ref_mask = BASELINES["контрпример: верхние 5 %"](X[:, :-1], names[:-1]).astype(bool) & oos
    ref_rate = ref_mask.sum() / max(int(oos.sum()), 1)
    print(f"Правило «верхние 5 %» срабатывает на {ref_rate*100:.1f} % дней.")
    print("Порог модели A зафиксирован на обучении (цель 13 %), фактическая частота ниже в таблице.\n")
    base = float(np.nanmean(y_case[oos]))
    print(f"{'правило / модель':<40}{'частота':>10}{'попадание':>12}{'lift':>8}")
    rows = []
    for bn, bf in BASELINES.items():
        f = bf(X[:, :-1], names[:-1]).astype(bool) & oos
        lf, _, n = lift(f, y_case)
        rows.append((bn, f.sum() / max(int(oos.sum()), 1), lf))
    validA = oos & ~np.isnan(score_A)
    fA = fires_A & oos          # порог из обучения, а не подогнанный под тест
    lfA, _, _ = lift(fA, y_case)
    rows.append(("МОДЕЛЬ A (обучена на метрике кейса)", fA.sum() / max(int(oos.sum()), 1), lfA))
    fB = fires_B & oos
    lfB, _, _ = lift(fB, y_case)
    rows.append(("МОДЕЛЬ B (обучена на метрике клиента)", fB.sum() / max(int(oos.sum()), 1), lfB))
    for nm, rate, lf in rows:
        star = "  * " if nm.startswith("МОДЕЛЬ") else "    "
        print(f"{star}{nm:<36}{rate*100:>9.1f}%{lf*base*100:>11.1f}%{lf:>8.2f}")
    mask = validA & ~np.isnan(y_case)
    if mask.sum() > 100:
        print(f"\n  AUC модели A по метрике кейса: {roc_auc_score(y_case[mask], score_A[mask]):.3f}")
        print(f"  Средний процентиль в момент срабатывания модели A: "
              f"{X[:, names.index('pct_range_90')][fA].mean():.0f} % "
              f"(правило срабатывает на 97 %)")

    # ═══ МЕТРИКА КЛИЕНТА ═══
    print("\n" + "=" * 104)
    print(f"МЕТРИКА КЛИЕНТА: выгода семьи против перевода в день зарплаты, окно {WINDOW} публикаций")
    print(f"Окон в тесте: {total_windows}. Потолок (оракул, знает будущее): {oracle:+.0f} бп")
    print("=" * 104)
    print(f"{'политика':<40}{'выгода':>12}{'95% ДИ':>20}{'сработало окон':>18}{'% оракула':>12}")

    def show(nm, fn, star=False):
        g, lo, hi, fr = policy_gain(fn)
        mark = "  * " if star else "    "
        print(f"{mark}{nm:<36}{g:>+11.0f}бп{f'[{lo:+.0f}; {hi:+.0f}]':>20}{fr*100:>17.0f}%{g/oracle*100:>11.0f}%")
        return g

    pct90 = X[:, names.index("pct_range_90")]
    show("Без продукта: перевод в день зарплаты", lambda c, i: True)
    show("Ждать до конца окна", lambda c, i: False)
    show("Правило: верхние 5 % диапазона", lambda c, i: pct90[row_of[(c, i)]] >= 95 if (c, i) in row_of else False)
    show("Правило: нижние 10 % диапазона", lambda c, i: pct90[row_of[(c, i)]] <= 10 if (c, i) in row_of else False)
    show("Правило: pct >= 85 (частота как у модели A)", lambda c, i: pct90[row_of[(c, i)]] >= 85 if (c, i) in row_of else False)
    show("МОДЕЛЬ A (метрика кейса)",
         lambda c, i: (r := row_of.get((c, i))) is not None and bool(fires_A[r]), True)
    show("МОДЕЛЬ B (метрика клиента)",
         lambda c, i: (r := row_of.get((c, i))) is not None and not np.isnan(score_B[r]) and score_B[r] > 0, True)
    print(f"    {'Оракул: лучший день окна':<36}{oracle:>+11.0f}бп{'—':>20}{'—':>18}{100:>11.0f}%")


if __name__ == "__main__":
    main()
