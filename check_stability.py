"""Устойчивость по годам. Среднее без разбивки лжёт.

Конфигурации берутся ДО теста — по внутренней валидации периода разработки.
Список, отобранный по результату на тесте, показывал бы устойчивость модели на
тех же годах, по которым она и выбрана. Такие строки тоже печатаются — для
сравнения — но помечены как справочные и результатом не являются.
"""
from __future__ import annotations

import numpy as np

from ml.data import CORRIDORS, REFERENCE, load
from ml.baselines import BASELINES
from ml.evaluate import REFERENCE_RULE, lift, reference_rate, train_cutoff
from ml.features import build_matrix
from ml.models import make_classifiers
from ml.selection import select_features, select_model
from ml.targets import benefit_forward_only, build_targets
from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds

s = load()
X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
dates = np.array([d for _, _, d in index], dtype=object)
Y = build_targets(s, index)
RATE = reference_rate(
    BASELINES[REFERENCE_RULE](X, names), dates, 2021)

# Выбор ДО теста: для каждого горизонта — модель с лучшим AUC на внутренней
# валидации периода разработки, отдельно на всех признаках и на отобранных.
CASES: list[tuple[int, str, bool, bool]] = []
for _h in (3, 5, 10):
    _y = Y[f"fav_h{_h}"]
    _cols, _, _ = select_features(X, _y, dates, names, 2021, horizon=_h, reach=target_reach_dates(index, s, _h))
    _n_all, _r_all = select_model(X, _y, dates, 2021, horizon=_h, reach=target_reach_dates(index, s, _h))
    _n_sel, _r_sel = select_model(X, _y, dates, 2021, cols=list(_cols), horizon=_h, reach=target_reach_dates(index, s, _h))
    if dict(_r_all)[_n_all] >= dict(_r_sel)[_n_sel]:
        CASES.append((_h, _n_all, False, True))
    else:
        CASES.append((_h, _n_sel, True, True))
# Справочно — то, что выигрывало НА ТЕСТЕ (результатом не является). Список
# вычисляется, а не вписан руками: захардкоженная строка молча устаревает при
# любом изменении данных или признаков и начинает лгать в сданном артефакте.
def _winner_on_test(h: int) -> tuple[str, bool]:
    """Конфигурация с лучшей достижимой выгодой НА ТЕСТЕ. Только для сравнения."""
    y = Y[f"fav_h{h}"]
    fwd = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        v = benefit_forward_only(s[c].values, i, h)
        if v is not None:
            fwd[r] = v
    sel_cols, _, _ = select_features(X, y, dates, names, 2021, horizon=h, reach=target_reach_dates(index, s, h))
    best = (-np.inf, "", False)
    reach = target_reach_dates(index, s, h)
    folds = list(walk_forward_folds(dates, 2021, h, reach=reach))
    for use_sel in (False, True):
        cols = list(sel_cols) if use_sel else list(range(X.shape[1]))
        for mname in make_classifiers():
            oos = np.zeros(len(y), bool)
            f = np.zeros(len(y), bool)
            for tr_i, te_i, _ in folds:
                tr = tr_i[~np.isnan(y[tr_i])]
                te = te_i[~np.isnan(y[te_i])]
                if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
                    continue
                oos[te] = True
                m = make_classifiers()[mname]
                m.fit(X[tr][:, cols], y[tr])
                f[te] = (m.predict_proba(X[te][:, cols])[:, 1]
                         >= train_cutoff(m.predict_proba(X[tr][:, cols])[:, 1], RATE))
            f &= oos
            g = float(np.nanmean(np.where(f, fwd, np.nan))) if f.sum() else -np.inf
            if g > best[0]:
                best = (g, mname, use_sel)
    return best[1], best[2]


for _h in (3, 5, 10):
    _nm, _sel = _winner_on_test(_h)
    CASES.append((_h, _nm, _sel, False))

print("=" * 100)
print("УСТОЙЧИВОСТЬ ПО ГОДАМ: достижимая выгода, бп (n — сработавших дней)")
print("=" * 100)
for h, mname, use_sel, chosen_before in CASES:
    y = Y[f"fav_h{h}"]
    fwd = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        b = benefit_forward_only(s[c].values, i, h)
        if b is not None:
            fwd[r] = b
    cols = list(range(X.shape[1]))
    if use_sel:
        cols, _, _ = select_features(X, y, dates, names, 2021, horizon=h, reach=target_reach_dates(index, s, h))
    oos = np.zeros(len(y), bool)
    sc = np.full(len(y), np.nan)
    f = np.zeros(len(y), bool)
    reach = target_reach_dates(index, s, h)
    for tr_i, te_i, _ in walk_forward_folds(dates, 2021, h, reach=reach):
        assert_no_overlap(dates, tr_i, te_i, h, index=index, series=s)
        tr = tr_i[~np.isnan(y[tr_i])]
        te = te_i[~np.isnan(y[te_i])]
        if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
            continue
        oos[te] = True
        m = make_classifiers()[mname]
        m.fit(X[tr][:, cols], y[tr])
        sc[te] = m.predict_proba(X[te][:, cols])[:, 1]
        # порог — по обучающим оценкам фолда
        f[te] = sc[te] >= train_cutoff(m.predict_proba(X[tr][:, cols])[:, 1], RATE)
    f = f & oos
    lf, _, _ = lift(f, y, scope=oos)
    tag = "отбор" if use_sel else "все"
    mark = "выбрано ДО теста" if chosen_before else "СПРАВОЧНО: выигрывало на тесте"
    print(f"\n  {mname} [{tag}], h={h}  |  lift {lf:.2f}, "
          f"достижимая {np.nanmean(np.where(f, fwd, np.nan)):+.0f} бп   ({mark})")
    yrs = sorted({d.year for d in dates[oos]})
    print("   " + "".join(f"{yy:>11}" for yy in yrs))
    r1 = r2 = ""
    pos = judged = 0
    for yy in yrs:
        msk = np.array([d.year == yy for d in dates]) & f & ~np.isnan(fwd)
        if msk.sum() > 15:
            val = np.nanmean(fwd[msk])
            r1 += f"{val:>+8.0f}бп"
            pos += val > 0
            judged += 1
        else:
            r1 += f"{'—':>11}"
        r2 += f"{'n=' + str(int(msk.sum())):>11}"
    print("   " + r1)
    print("   " + r2)
    # Знаменатель — годы, по которым вообще есть оценка. «1 из 6» при четырёх
    # прочерках читается как «пять лет отрицательны», хотя четыре из них просто
    # не измерены: у редкой модели в году меньше 16 срабатываний.
    skipped = len(yrs) - judged
    tail = f"  (ещё {skipped} года без оценки: <16 срабатываний)" if skipped else ""
    print(f"   лет с положительным результатом: {pos} из {judged} оценённых{tail}")

    # Сколько итог держится на худшем годе. «Пять лет из шести в плюсе» ничего не
    # говорит о величине провала: год с 21 срабатыванием и год с 339 весят в общей
    # цифре по-разному, и разобрать это надо самим, до вопроса с защиты.
    per_year = [(yy, np.nanmean(fwd[m]), int(m.sum()))
                for yy in yrs
                for m in [np.array([d.year == yy for d in dates]) & f & ~np.isnan(fwd)]
                if m.sum() > 15]
    if len(per_year) >= 2:
        n_tot = sum(n for _y, _v, n in per_year)
        wmean = sum(v * n for _y, v, n in per_year) / n_tot
        umean = sum(v for _y, v, _n in per_year) / len(per_year)
        wy, wv, wn = min(per_year, key=lambda r: r[1])
        rest = [(y, v, n) for y, v, n in per_year if y != wy]
        wo = sum(v * n for _y, v, n in rest) / sum(n for _y, _v, n in rest)
        print(f"   средневзвешенное по числу срабатываний {wmean:+.0f} бп, "
              f"невзвешенное по годам {umean:+.0f} бп")
        print(f"   худший год {wy}: {wv:+.0f} бп при n={wn} "
              f"({wn / n_tot * 100:.0f} % срабатываний); без него {wo:+.0f} бп, "
              f"его вклад в итог {wmean - wo:+.0f} бп")
