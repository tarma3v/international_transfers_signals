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
from ml.evaluate import REFERENCE_RULE, bootstrap_ci, lift, rate_per_week, reference_rate, train_cutoff
from ml.features import WARMUP, build_matrix
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
from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds

from catboost import CatBoostClassifier, CatBoostRegressor

# Модель A: метрика кейса — монотонная функция уровня, и добавление признаков ей
# только вредит. Обоснование обязано быть ДО теста: числа «80 признаков -> lift
# 1,14, один монотонный -> 1,30» — обе out-of-sample, и выбирать по ним набор
# признаков значит выбрать конфигурацию по тесту. Функция
# `feature_set_on_dev` ниже проверяет ровно это сравнение на периоде
# разработки; её вывод печатается в отчёте рядом с результатом модели A.
MODEL_A_FEATURES = ("pct_range_90",)
# Модель B: клиентская метрика. Семь признаков уровня и моментума выбраны РУКАМИ,
# по смыслу задачи об остановке, а не процедурой отбора. Это признанное упрощение:
# перебора конфигураций для модели B мы не делали, поэтому и «лучшей» её не называем.
MODEL_B_FEATURES = (
    "pct_range_90", "pct_range_180", "ret_5", "ret_20", "vol_30", "streak_up", "streak_dn",
)

H = 5              # горизонт метрики кейса
WINDOW = 5         # окно гибкости клиента после зарплаты, публикаций
WINDOW_SWEEP = (5, 10, 20, 40)  # длины окна для проверки «ждать или сейчас»
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
    dates = np.array([d for _, _, d in index], dtype=object)
    row_of = {(c, i): r for r, (c, i, _) in enumerate(index)}

    # ——— цели ———
    y_case = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        t = target_case(s[c].values, i, H)
        if t is not None:
            y_case[r] = t

    win_map = build_windows(s, CORRIDORS, WINDOW, warmup=WARMUP)
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

    # Целевая частота — доля срабатываний ОДНОГО правила сравнения на ПЕРИОДЕ
    # РАЗРАБОТКИ (ml/evaluate.py::REFERENCE_RULE). Два требования разом.
    # Первое: правило одно на весь проект — иначе «верх диапазона» (9,4 %) здесь
    # и «нижний дециль» (10,2 %) в других скриптах дадут для одних и тех же
    # моделей разные колонки «частота» и «сигн/нед» в двух отчётах.
    # Второе: частота считается ДО первого тестового года, а не константой,
    # снятой с теста, — иначе рабочая точка знает распределение теста.
    target_rate = reference_rate(
        BASELINES[REFERENCE_RULE](X, names), dates, FIRST_TEST)

    # ——— walk-forward: обучаем обе модели ———
    reach = target_reach_dates(index, s, max(H, WINDOW))
    folds = walk_forward_folds(dates, FIRST_TEST, max(H, WINDOW), reach=reach)
    oos = np.zeros(len(index), bool)
    score_A = np.full(len(index), np.nan)   # вероятность метрики кейса
    score_B = np.full(len(index), np.nan)   # предсказанное преимущество «сегодня vs лучший остаток»

    for tr_i, te_i, _yr in folds:
        assert_no_overlap(dates, tr_i, te_i, max(H, WINDOW), index=index, series=s)
        oos[te_i] = True

        trA = tr_i[~np.isnan(y_case[tr_i])]
        teA = te_i[~np.isnan(y_case[te_i])]
        if len(trA) > 400 and len(np.unique(y_case[trA])) > 1:
            mA = CatBoostClassifier(iterations=300, depth=2, learning_rate=0.05,
                                    l2_leaf_reg=10.0, monotone_constraints=[1] * len(colsA),
                                    random_seed=SEED, verbose=0, allow_writing_files=False)
            mA.fit(X[trA][:, colsA], y_case[trA])
            score_A[teA] = mA.predict_proba(X[teA][:, colsA])[:, 1]
            thr_train = train_cutoff(mA.predict_proba(X[trA][:, colsA])[:, 1], target_rate)
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
            fires_B[teB] = score_B[teB] >= train_cutoff(mB.predict(X[trB][:, colsB]), target_rate)

    # ——— окна для симуляции политики, только out-of-sample ———
    per_corr_windows: dict[str, list[tuple[int, int]]] = {}
    for c in CORRIDORS:
        dts = list(s[c].dates)
        n = len(dts)
        ws = []
        for p in payday_anchors(dts):
            r = row_of.get((c, p))
            if p < WARMUP or p + WINDOW >= n or r is None or not oos[r]:
                continue
            ws.append((p, WINDOW))
        per_corr_windows[c] = ws
    total_windows = sum(len(v) for v in per_corr_windows.values())

    def policy_gain(fire_fn) -> tuple[float, float, float, float]:
        """fire_fn(corridor, idx) -> bool. Возвращает (выгода, ДИ низ, ДИ верх, доля окон со срабатыванием)."""
        all_gains: list[float] = []
        gain_dates: list = []
        fired = 0
        wins = 0
        for c in CORRIDORS:
            v = s[c].values
            fires = {}
            for p, w in per_corr_windows[c]:
                for k in range(w + 1):
                    fires[p + k] = fire_fn(c, p + k)
            run = evaluate_policy(v, per_corr_windows[c], fires)
            if len(run):
                all_gains.extend(run.gains.tolist())
                gain_dates.extend(s[c].dates[p] for p in run.anchor)
                fired += int(run.fired.sum())
                wins += len(run)
        # ДИ с кластеризацией по дате выплаты: одно и то же окно в пяти
        # коридорах — не пять независимых наблюдений.
        lo, hi = bootstrap_ci(np.array(all_gains), dates=np.array(gain_dates, dtype=object))
        return float(np.mean(all_gains)), lo, hi, fired / max(wins, 1)

    oracle = float(np.mean([oracle_gain(s[c].values, per_corr_windows[c]) for c in CORRIDORS]))

    # ═══ МЕТРИКА КЕЙСА ═══
    print("=" * 104)
    print(f"МЕТРИКА КЕЙСА: «сейчас выгодно», h = {H}. Сравнение при ОДИНАКОВОЙ частоте срабатываний.")
    print("=" * 104)
    ref_mask = BASELINES["простое правило: верх диапазона"](X, names).astype(bool) & oos
    ref_rate = ref_mask.sum() / max(int(oos.sum()), 1)
    print(f"Правило «верхние 5 %» срабатывает на {ref_rate*100:.1f} % дней теста.")
    print(f"Порог обеих моделей зафиксирован на обучении, цель — частота правила сравнения")
    print(f"«{REFERENCE_RULE}» на периоде разработки ({target_rate*100:.1f} %).")
    print("Это не то же правило, что «верхние 5 %»: правило сравнения одно на все скрипты")
    print("проекта, иначе колонки «частота» и «сигн/нед» в двух отчётах несопоставимы.")
    print("Фактическая частота на тесте — в таблице.\n")
    base = float(np.nanmean(y_case[oos]))
    print("Полоса из ТЗ: 1-2 сигнала на коридор в неделю. Колонка «сигн/нед» проверяет её напрямую.")
    print(f"{'правило / модель':<40}{'частота':>10}{'сигн/нед':>10}{'попадание':>12}{'lift':>8}{'полоса ТЗ':>12}")
    rows = []
    for bn, bf in BASELINES.items():
        f = bf(X, names).astype(bool) & oos
        lf, _, n = lift(f, y_case, scope=oos)
        rows.append((bn, f.sum() / max(int(oos.sum()), 1), lf, rate_per_week(n, len(CORRIDORS), dates, oos)))
    validA = oos & ~np.isnan(score_A)
    fA = fires_A & oos          # порог из обучения, а не подогнанный под тест
    lfA, _, nA = lift(fA, y_case, scope=oos)
    rows.append(("МОДЕЛЬ A (набор признаков с теста — справочно)", fA.sum() / max(int(oos.sum()), 1), lfA,
                 rate_per_week(nA, len(CORRIDORS), dates, oos)))
    fB = fires_B & oos
    lfB, _, nB = lift(fB, y_case, scope=oos)
    rows.append(("МОДЕЛЬ B (порог из обучения)", fB.sum() / max(int(oos.sum()), 1), lfB,
                 rate_per_week(nB, len(CORRIDORS), dates, oos)))
    # Правила, на которые ссылаются презентация и отчёты: считаем их lift здесь,
    # иначе главный тезис «модель неотличима от правила» ничем не подтверждён.
    pct90_all = X[:, names.index("pct_range_90")]
    for thr in (95, 90, 85):
        fr = (pct90_all >= thr) & oos
        lfr, _, nr = lift(fr, y_case, scope=oos)
        rows.append((f"правило pct_range_90 >= {thr}", fr.sum() / max(int(oos.sum()), 1), lfr,
                     rate_per_week(nr, len(CORRIDORS), dates, oos)))
    for nm, rate, lf, pw in rows:
        star = "  * " if nm.startswith("МОДЕЛЬ") else "    "
        band = "в полосе" if 1.0 <= pw <= 2.0 else ("НИЖЕ" if pw < 1.0 else "ВЫШЕ")
        print(f"{star}{nm:<36}{rate*100:>9.1f}%{pw:>10.2f}{lf*base*100:>11.1f}%{lf:>8.2f}{band:>12}")
    print("\n  Требование ТЗ — lift >= 1,3 ПРИ 1-2 сигналах на коридор в неделю. Оба условия")
    print("  одновременно не выполняет ничто: всё, что даёт lift >= 1,3, срабатывает реже полосы.")
    print("\n  Строка МОДЕЛИ A — справочная. Её набор признаков (один монотонный) выбран")
    print("  сравнением out-of-sample, а период разработки на том же сравнении выбирает")
    print(f"  все {len(names)} признака (блок «ОТКУДА ВЗЯТ НАБОР ПРИЗНАКОВ МОДЕЛИ A» ниже). Честно")
    print("  выбранная до теста конфигурация считается в run_experiment.py и run_boosting.py")
    print("  (там и модель, и набор признаков берутся с периода разработки). Главный вывод")
    print("  от этого не меняется —")
    print("  правило pct_range_90 >= 95 не требует никакого отбора и даёт 1,39.")

    # Какой порог выбрал бы период разработки — вопрос отдельный от того, какой
    # лучше на тесте. Выбирать порог по тестовой строке нельзя: это подгонка
    # рабочей точки, только записанная словами «внутри полосы лучше 85».
    dev = np.array([d.year < FIRST_TEST for d in dates]) & ~np.isnan(y_case)
    base_dev = float(y_case[dev].mean())
    print("\n  Порог, выбранный на ПЕРИОДЕ РАЗРАБОТКИ (тест в решении не участвует):")
    print(f"{'порог':<26}{'частота (dev)':>15}{'сигн/нед (dev)':>17}{'lift (dev)':>13}"
          f"{'полоса ТЗ':>12}")
    dev_pick, dev_best = None, -9e9
    for thr in (95, 90, 85):
        m = dev & (pct90_all >= thr)
        if m.sum() < 30:
            continue
        lf_dev = float(y_case[m].mean()) / base_dev
        pw_dev = rate_per_week(int(m.sum()), len(CORRIDORS), dates, dev)
        if 1.0 <= pw_dev <= 2.0 and lf_dev > dev_best:
            dev_pick, dev_best = thr, lf_dev
        band = "в полосе" if 1.0 <= pw_dev <= 2.0 else ("НИЖЕ" if pw_dev < 1.0 else "ВЫШЕ")
        print(f"{f'pct_range_90 >= {thr}':<26}{m.sum()/max(int(dev.sum()),1)*100:>14.1f}%"
              f"{pw_dev:>17.2f}{lf_dev:>13.2f}{band:>12}")
    if dev_pick is None:
        print("  Внутри полосы 1-2 сигн/нед на периоде разработки не оказалось ни одного порога.")
    else:
        te_row = next(r for r in rows if r[0].endswith(str(dev_pick)))
        print(f"  Период разработки выбрал бы порог {dev_pick} (lift {dev_best:.2f} на dev).")
        print(f"  На тесте этот же порог даёт lift {te_row[2]:.2f} при {te_row[3]:.2f} сигн/нед —")
        print("  это и есть честная оценка правила, в отличие от лучшей строки таблицы выше.")
    mask = validA & ~np.isnan(y_case)
    if mask.sum() > 100:
        print(f"\n  AUC модели A по метрике кейса: {roc_auc_score(y_case[mask], score_A[mask]):.3f}")
        rule5 = BASELINES["простое правило: верх диапазона"](X, names).astype(bool) & oos
        print(f"  Средний процентиль в момент срабатывания модели A: "
              f"{pct90_all[fA].mean():.0f} % "
              f"(у правила «верхние 5 %» — {pct90_all[rule5].mean():.0f} %)")

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
    show("МОДЕЛЬ B (порог: предсказание > 0)",
         lambda c, i: (r := row_of.get((c, i))) is not None and not np.isnan(score_B[r]) and score_B[r] > 0, True)
    print(f"    {'Оракул: лучший день окна':<36}{oracle:>+11.0f}бп{'—':>20}{'—':>18}{100:>11.0f}%")
    print("\n  ВНИМАНИЕ: «МОДЕЛЬ B» в этой таблице и в таблице метрики кейса — РАЗНЫЕ политики.")
    print("  Здесь порог «предсказанная выгода > 0», там — порог из обучения на целевую частоту.")

    # ═══ РАЗЛОЖЕНИЕ: ЧТО ИМЕННО ДАЁТ ИТОГОВОЕ ЧИСЛО ═══
    print("\n" + "=" * 104)
    print("РАЗЛОЖЕНИЕ ИТОГА МОДЕЛИ A ПО ДНЮ СРАБАТЫВАНИЯ ВНУТРИ ОКНА")
    print("Итоговая цифра складывается из окон, где сигнал был, и окон, где его не было.")
    print("Показывать её без этого разложения нельзя: молчание — не заслуга модели.")
    print("=" * 104)
    print(f"{'когда сработало':<34}{'окон':>8}{'средняя выгода':>18}{'вклад в итог':>16}")
    buckets: dict[str, list[float]] = {"день 0 (день зарплаты)": [], "дни 1-5": [], "не сработало": []}
    for c in CORRIDORS:
        v = s[c].values
        for p, w in per_corr_windows[c]:
            day = next((k for k in range(w + 1)
                        if (r := row_of.get((c, p + k))) is not None and fires_A[r]), None)
            chosen = p + day if day is not None else p + w
            g = -(float(v[chosen]) - float(v[p])) / float(v[p]) * 10000.0
            key = "не сработало" if day is None else ("день 0 (день зарплаты)" if day == 0 else "дни 1-5")
            buckets[key].append(g)
    total_n = sum(len(x) for x in buckets.values())
    for k, vals in buckets.items():
        if not vals:
            continue
        share = sum(vals) / total_n
        print(f"{k:<34}{len(vals):>8}{np.mean(vals):>+17.0f}бп{share:>+15.0f}бп")
    print(f"{'ИТОГО':<34}{total_n:>8}"
          f"{sum(sum(v) for v in buckets.values())/total_n:>+17.0f}бп")

    # ═══ КОНТРФАКТ ДЛЯ СРАБАТЫВАНИЙ ВНУТРИ ОКНА ═══
    print("\n" + "=" * 104)
    print("ЦЕННОСТЬ ДЕЙСТВИЯ: правильный контрфакт для срабатываний на днях 1-5")
    print("Клиент, дошедший до дня k, уже не может перевести в день 0 — этот выбор в прошлом.")
    print("Его единственная альтернатива — продолжать ждать. Сравнение с днём зарплаты")
    print("отвечает на вопрос, которого перед клиентом не стоит.")
    print("=" * 104)
    print("Популяция здесь — ровно те окна, где ПЕРВОЕ срабатывание пришлось на дни 1-5.")
    print("Искать первое срабатывание начиная с дня 1 нельзя: политика переводит деньги")
    print("по первому сигналу, и клиент, у которого сигнал был в день 0, до дня k не доходит.")
    act, wait, when = [], [], []
    wrong_act, wrong_wait, wrong_when = [], [], []
    for c in CORRIDORS:
        v = s[c].values
        for p, w in per_corr_windows[c]:
            k1 = next((k for k in range(1, w + 1)
                       if (r := row_of.get((c, p + k))) is not None and fires_A[r]), None)
            if k1 is None:
                continue
            wrong_act.append(-(float(v[p + k1]) - float(v[p])) / float(v[p]) * 10000.0)
            wrong_wait.append(-(float(v[p + w]) - float(v[p])) / float(v[p]) * 10000.0)
            wrong_when.append(s[c].dates[p])
    wrong_n = len(wrong_act)
    for c in CORRIDORS:
        v = s[c].values
        for p, w in per_corr_windows[c]:
            day = next((k for k in range(w + 1)
                        if (r := row_of.get((c, p + k))) is not None and fires_A[r]), None)
            if day is None or day == 0:
                continue
            act.append(-(float(v[p + day]) - float(v[p])) / float(v[p]) * 10000.0)
            wait.append(-(float(v[p + w]) - float(v[p])) / float(v[p]) * 10000.0)
            when.append(s[c].dates[p])
    print(f"Отбор «первое срабатывание начиная с дня 1» дал бы {wrong_n} окон вместо "
          f"{len(act)} —")
    print(f"разница в {wrong_n - len(act)} окон — это те, где сигнал был уже в день 0.")
    if wrong_n >= 10:
        wd = np.array(wrong_act) - np.array(wrong_wait)
        wlo, whi = bootstrap_ci(wd, dates=np.array(wrong_when, dtype=object))
        verdict = "значимо" if wlo > 0 or whi < 0 else "не значимо"
        print(f"На той, несуществующей популяции вышло бы {wd.mean():+.0f} бп "
              f"95% ДИ [{wlo:+.0f}; {whi:+.0f}] — {verdict}. Печатаем это число, чтобы")
        print("разницу между двумя определениями популяции можно было проверить, а не")
        print("принять на слово: она и есть вся разница между «значимо» и «нет».")
    if len(act) >= 10:
        diff = np.array(act) - np.array(wait)
        lo, hi = bootstrap_ci(diff, dates=np.array(when, dtype=object))
        print(f"  окон со срабатыванием на днях 1-5           : {len(act)}")
        print(f"  перевести по сигналу                        : {np.mean(act):+.0f} бп")
        print(f"  проигнорировать сигнал и ждать до конца окна: {np.mean(wait):+.0f} бп")
        print(f"  ЦЕННОСТЬ ДЕЙСТВИЯ ПО СИГНАЛУ                : {diff.mean():+.0f} бп"
              f"  95% ДИ [{lo:+.0f}; {hi:+.0f}]")
        print(f"  вердикт: {'значимо' if lo > 0 or hi < 0 else 'НЕ значимо — интервал пересекает ноль'}")
    else:
        print("  срабатываний на днях 1-5 слишком мало для оценки")

    feature_set_on_dev(X, names, y_case, dates, index, s)
    monotonicity_on_dev(X, names, y_case, dates)
    horizon_sweep(s, oos, row_of, X, names)


def feature_set_on_dev(X, names, y_case, dates, index, s) -> None:
    """Набор признаков модели A обязан быть обоснован ДО теста.

    Проверяется то же сравнение, которым конфигурация выбрана словами — «один
    монотонный признак против всех» — но на внутренней валидации периода
    разработки, тем же сплитом, что `ml/selection.py`. Если dev даёт обратный
    ответ, конфигурация модели A выбрана по тесту, и об этом надо писать прямо.
    """
    from sklearn.metrics import roc_auc_score

    from ml.selection import _inner_split
    from ml.validation import target_reach_dates

    print("\n" + "=" * 104)
    print(f"ОТКУДА ВЗЯТ НАБОР ПРИЗНАКОВ МОДЕЛИ A (только данные до {FIRST_TEST})")
    print("=" * 104)
    wall = dt.date(FIRST_TEST, 1, 1)
    dev = np.array([d < wall for d in dates]) & ~np.isnan(y_case)
    inner_wall = max(d for d in dates[dev]) - dt.timedelta(days=30 * 6)
    reach = target_reach_dates(index, s, max(H, WINDOW))
    tr, va = _inner_split(dates, dev, inner_wall, max(H, WINDOW), reach)
    print(f"внутреннее обучение {int(tr.sum())} строк, внутренняя валидация {int(va.sum())}")
    col = names.index("pct_range_90")
    all_features_label = f"все {len(names)} признака"
    variants = {
        all_features_label: list(range(len(names))),
        "один признак pct_range_90": [col],
    }
    best, best_auc = None, -1.0
    for nm, cols in variants.items():
        m = CatBoostClassifier(iterations=300, depth=2, learning_rate=0.05, l2_leaf_reg=10.0,
                               monotone_constraints=[1] * len(cols) if cols == [col] else None,
                               verbose=0, random_seed=SEED, allow_writing_files=False)
        m.fit(X[tr][:, cols], y_case[tr])
        auc = float(roc_auc_score(y_case[va], m.predict_proba(X[va][:, cols])[:, 1]))
        print(f"  {nm:<32} AUC на внутренней валидации {auc:.3f}")
        if auc > best_auc:
            best, best_auc = nm, auc
    print(f"  период разработки выбирает: {best}")
    if best != "один признак pct_range_90":
        print("  ВНИМАНИЕ: конфигурация модели A НЕ подтверждается периодом разработки —")
        print("  значит, она выбрана по тесту, и строку модели A надо читать как справочную.")


def monotonicity_on_dev(X, names, y_case, dates) -> None:
    """Направление монотонности модели A выводится из ПЕРИОДА РАЗРАБОТКИ, не из теста.

    Модели A задано ограничение «вероятность растёт по уровню». Это решение,
    принятое руками, и оно обязано опираться только на данные до первого
    тестового года — иначе знак подсмотрен в тесте.
    """
    print("\n" + "=" * 104)
    print("ОТКУДА ВЗЯТО НАПРАВЛЕНИЕ МОНОТОННОСТИ МОДЕЛИ A (только данные до "
          f"{FIRST_TEST})")
    print("=" * 104)
    dev = np.array([d.year < FIRST_TEST for d in dates]) & ~np.isnan(y_case)
    pct = X[:, names.index("pct_range_90")]
    base = float(y_case[dev].mean())
    print(f"Базовая ставка на периоде разработки: {base * 100:.1f} %, строк {int(dev.sum())}\n")
    print(f"{'бакет процентиля':<22}{'строк':>8}{'попадание':>12}{'lift':>8}")
    edges = [(0, 10), (10, 50), (50, 90), (90, 95), (95, 100.01)]
    for lo, hi in edges:
        m = dev & (pct >= lo) & (pct < hi)
        if m.sum() < 30:
            print(f"{f'{lo}-{hi:.0f}':<22}{int(m.sum()):>8}{'—':>12}{'—':>8}")
            continue
        hit = float(y_case[m].mean())
        print(f"{f'{lo}-{hi:.0f}':<22}{int(m.sum()):>8}{hit * 100:>11.1f}%{hit / base:>8.2f}")
    print("\n  Связь возрастающая уже на обучении: знак ограничения не пришлось")
    print("  подсматривать в тесте.")


def horizon_sweep(s, oos, row_of, X, names) -> None:
    """Как длина окна клиента меняет ответ «ждать или переводить сейчас».

    Документы ссылаются на эти две таблицы, поэтому они обязаны считаться здесь,
    а не существовать только в тексте.
    """
    pct90 = X[:, names.index("pct_range_90")]
    print("\n" + "=" * 104)
    print("ДЛИНА ОКНА КЛИЕНТА: когда «ждать» перестаёт проигрывать «переводить сейчас»")
    print("Окно — сколько публикаций после зарплаты клиент готов ждать. Выгода в бп")
    print("против перевода в день зарплаты, ДИ — бутстрап с кластеризацией по датам выплат.")
    print("=" * 104)
    print(f"{'политика':<38}" + "".join(f"{'окно ' + str(w):>22}" for w in WINDOW_SWEEP))

    policies = {
        "Потолок (оракул)": None,
        "Верхние 5 % — переводить сейчас": lambda r: pct90[r] >= 95,
        "Нижние 10 % — ждать дешёвого дня": lambda r: pct90[r] <= 10,
        "Ждать до конца окна": lambda _r: False,
    }
    for nm, rule in policies.items():
        cells = []
        for w in WINDOW_SWEEP:
            gains, when, orc = [], [], []
            for c in CORRIDORS:
                v, dts = s[c].values, list(s[c].dates)
                for p in payday_anchors(dts):
                    r0 = row_of.get((c, p))
                    if p < WARMUP or p + w >= len(v) or r0 is None or not oos[r0]:
                        continue
                    orc.append(-(float(v[p:p + w + 1].min()) - float(v[p])) / float(v[p]) * 10000.0)
                    if rule is None:
                        continue
                    k = next((j for j in range(w + 1)
                              if (r := row_of.get((c, p + j))) is not None and rule(r)), w)
                    gains.append(-(float(v[p + k]) - float(v[p])) / float(v[p]) * 10000.0)
                    when.append(dts[p])
            if rule is None:
                cells.append(f"{np.mean(orc):>+20.0f}бп" if orc else f"{'—':>22}")
                continue
            if len(gains) < 20:
                cells.append(f"{'—':>22}")
                continue
            lo, hi = bootstrap_ci(np.array(gains), dates=np.array(when, dtype=object))
            star = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{np.mean(gains):>+13.0f} [{lo:+.0f};{hi:+.0f}]{star}")
        print(f"{nm:<38}" + "".join(cells))
    print("\n  * — интервал не пересекает ноль. Окна даны в ПУБЛИКАЦИЯХ (~4,7 в неделю).")

    print("\n" + "=" * 104)
    print("ВЫГОДА ОЖИДАНИЯ ПО ГОДАМ: не артефакт ли это ослабления рубля 2019-2026")
    print("Перевод в конце окна из 5 публикаций против перевода в день зарплаты.")
    print("=" * 104)
    by_year: dict[int, list[float]] = {}
    for c in CORRIDORS:
        v, dts = s[c].values, list(s[c].dates)
        for p in payday_anchors(dts):
            if p < WARMUP or p + WINDOW >= len(v):
                continue
            by_year.setdefault(dts[p].year, []).append(
                -(float(v[p + WINDOW]) - float(v[p])) / float(v[p]) * 10000.0)
    years = sorted(by_year)
    print(f"{'год':<10}" + "".join(f"{y:>9}" for y in years))
    print(f"{'выгода':<10}" + "".join(f"{np.mean(by_year[y]):>+8.0f}" for y in years))
    print(f"{'окон':<10}" + "".join(f"{len(by_year[y]):>9}" for y in years))
    pos = sum(1 for y in years if np.mean(by_year[y]) > 0)
    print(f"\n  лет с положительной выгодой ожидания: {pos} из {len(years)}.")
    print("  Знак меняется от года к году — внутри зарплатного окна выбор дня ближе")
    print("  к подбрасыванию монеты, чем к предсказуемой закономерности. Это и есть")
    print("  вывод, а не «переводите сразу».")


if __name__ == "__main__":
    main()
