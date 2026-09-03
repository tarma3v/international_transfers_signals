"""Сводная таблица по всем горизонтам + важность признаков. Пишет results.md."""
from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import REFERENCE_RULE, UPLIFT_COMPARATOR, bootstrap_ci, lift, rate_per_week, reference_rate, train_cutoff
from ml.features import build_matrix
from ml.models import make_classifiers
from ml.selection import select_model
from ml.targets import HORIZONS, benefit_forward_only, build_targets
from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds


s = load()
X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
dates = np.array([d for _, _, d in index], dtype=object)
Y = build_targets(s, index)
RATE = reference_rate(
    BASELINES[REFERENCE_RULE](X, names), dates, 2021)

lines: list[str] = []
W = lines.append
W("# Результаты: признаки без заглядывания, walk-forward, uplift к индикаторам ТЗ\n")
W(f"Строк {X.shape[0]}, признаков {X.shape[1]}, коридоров {len(CORRIDORS)}, "
  f"период {min(dates)}..{max(dates)}.\n")
W("Обучение — расширяющееся окно, тест — календарный год с 2021. Очистка считается\n"
  "в ПУБЛИКАЦИЯХ, а не в календарных днях: горизонт задан в публикациях, и запас,\n"
  "отсчитанный в днях, оставил бы в обучении строки, чья цель лежит внутри теста.")
W("«Достижимая выгода» — форвардная половина метрики заказчика: то, что клиент реально может забрать.\n")
W("## Методология\n")
W("Пять решений фиксируются ДО теста, иначе тест участвует в них и результат завышен:\n")
W("* **рабочая точка** — порог срабатывания считается по обучающим оценкам каждого фолда")
W("  (`ml/evaluate.py::train_cutoff`), а не подгоняется под частоту правила на тесте.")
W("  Поэтому частоты у строк в таблицах разные;")
W("* **модель** для строки uplift выбирается по AUC на внутренней валидации периода")
W("  разработки (до 2021) — `ml/selection.py::select_model`. Взять лучшую по тесту нельзя:")
W("  разрыв к ТЗ раздувается тем сильнее, чем больше перебор. Для сравнения рядом")
W("  печатается и максимум по тесту — он результатом не является;")
W("* **очистка** — запас между обучением и тестом считается в публикациях")
W("  (`ml/validation.py::target_reach_dates`), а сторож `assert_no_overlap` пересчитывает")
W("  достижимые даты САМ, независимым кодом: получить тот же массив, которым сделана")
W("  очистка, значило бы сравнить ошибку с самой собой;")
W("* **компаратор для uplift** назван до теста (`ml/evaluate.py::UPLIFT_COMPARATOR`).")
W("  Максимум по четырём индикаторам, взятый с теста, — max-статистика, а не парная")
W("  оценка: у неё нет распределения и она не воспроизводится на новых данных. Она")
W("  печатается отдельной строкой и результатом не является;")
W("* **знаменатель lift** — базовая ставка считается на тех же строках, на которых")
W("  оценивается сигнал (`lift(..., scope=oos)`), иначе дрейф режима завышает lift.\n")
W("Все доверительные интервалы — бутстрап с ресемплингом ДНЕЙ целиком: выгода в пяти")
W("коридорах в один день коррелирована на 0,86, и построчный интервал занижен вдвое.\n")

for tname, tlabel in (("fav", "Сейчас выгодно"), ("close", "Окно закрывается")):
    W(f"\n## Цель: {tlabel}\n")
    W("| h | правило / модель | частота | сигн/нед | lift | выгода ±h | **достижимая** | 95% ДИ |")
    W("|---|---|---|---|---|---|---|---|")
    for h in HORIZONS:
        y = Y[f"{tname}_h{h}"]
        ben = Y[f"benefit_h{h}"]
        fwd = np.full(len(index), np.nan)
        for r, (c, i, _) in enumerate(index):
            b = benefit_forward_only(s[c].values, i, h)
            if b is not None:
                fwd[r] = b
        reach = target_reach_dates(index, s, h)
        folds = walk_forward_folds(dates, 2021, h, reach=reach)
        oos = np.zeros(len(y), bool)
        sc = {m: np.full(len(y), np.nan) for m in make_classifiers()}
        fr = {m: np.zeros(len(y), bool) for m in make_classifiers()}
        for tr_i, te_i, _ in folds:
            assert_no_overlap(dates, tr_i, te_i, h, index=index, series=s)
            tr = tr_i[~np.isnan(y[tr_i])]
            te = te_i[~np.isnan(y[te_i])]
            if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
                continue
            oos[te] = True
            for m, mdl in make_classifiers().items():
                mdl.fit(X[tr], y[tr])
                sc[m][te] = mdl.predict_proba(X[te])[:, 1]
                fr[m][te] = sc[m][te] >= train_cutoff(mdl.predict_proba(X[tr])[:, 1], RATE)
        if oos.sum() < 200:
            continue
        rows = []
        for bn, bf in BASELINES.items():
            f = bf(X, names).astype(bool) & oos
            rows.append((bn, f, False))
        for m in sc:
            rows.append((m, fr[m] & oos, True))
        chosen, _rep = select_model(X, y, dates, 2021, horizon=h, reach=target_reach_dates(index, s, h))
        best = {"tz": -9e9, "md": -9e9, "sel": float("nan"), "cmp": float("nan")}
        for nm, f, is_md in rows:
            lf, _, n = lift(f, y, scope=oos)
            rate = f.sum() / max(int(oos.sum()), 1)
            pw = rate_per_week(n, len(CORRIDORS), dates, oos)
            sym = float(np.nanmean(np.where(f, ben, np.nan)))
            fw = float(np.nanmean(np.where(f, fwd, np.nan)))
            sel = f & ~np.isnan(fwd)
            lo, hi = bootstrap_ci(fwd[sel], dates=dates[sel])
            tag = "**" if is_md else ""
            W(f"| {h} | {tag}{nm}{tag} | {rate*100:.0f}% | {pw:.2f} | {lf:.2f} | "
              f"{sym:+.0f} бп | **{fw:+.0f} бп** | [{lo:+.0f}; {hi:+.0f}] |")
            if is_md:
                best["md"] = max(best["md"], fw)
                if nm == chosen:
                    best["sel"] = fw
            elif nm.startswith("ТЗ:"):
                best["tz"] = max(best["tz"], fw)
                if nm == UPLIFT_COMPARATOR:
                    best["cmp"] = fw
        W(f"| {h} | *uplift к назначенному ДО теста компаратору «{UPLIFT_COMPARATOR}»; "
          f"модель тоже выбрана до теста ({chosen})* | | | | | "
          f"**{best['sel']-best['cmp']:+.0f} бп** | |")
        W(f"| {h} | *справочно — к максимуму по тесту среди четырёх индикаторов "
          f"(max-статистика, не парная оценка)* | | | | | "
          f"{best['sel']-best['tz']:+.0f} бп | |")
        W(f"| {h} | *справочно — максимум по тесту среди моделей "
          f"(результатом не является)* | | | | | "
          f"{best['md']-best['tz']:+.0f} бп | |")

# важность признаков на самой сильной конфигурации
y = Y["fav_h5"]
folds = walk_forward_folds(dates, 2021, 5, reach=target_reach_dates(index, s, 5))
tr_i, te_i, _ = folds[-1]
tr = tr_i[~np.isnan(y[tr_i])]
te = te_i[~np.isnan(y[te_i])]
mdl = make_classifiers()["случайный лес"]
mdl.fit(X[tr], y[tr])
# scoring обязателен: по умолчанию берётся accuracy, а при базовой ставке 29 %
# модель почти всегда предсказывает «нет» и accuracy не реагирует на перестановки.
imp = permutation_importance(
    mdl, X[te], y[te], scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1
)
order = np.argsort(imp.importances_mean)[::-1][:15]
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y[te], mdl.predict_proba(X[te])[:, 1])
W("\n\n## Важность признаков (перестановка, падение ROC AUC, случайный лес, «сейчас выгодно» h=5)\n")
W(f"AUC модели на последнем фолде: {auc:.3f} (0,5 = нет информации)\n")
W("| признак | вклад |")
W("|---|---|")
for j in order:
    W(f"| `{names[j]}` | {imp.importances_mean[j]:+.4f} |")

open("results/results.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines[-20:]))
print("\n>>> results.md записан")
