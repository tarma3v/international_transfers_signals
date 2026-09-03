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
from ml.evaluate import lift, reference_rate, train_cutoff
from ml.features import build_matrix
from ml.models import make_classifiers
from ml.selection import select_features, select_model
from ml.targets import benefit_forward_only, build_targets
from ml.validation import assert_no_overlap, walk_forward_folds

s = load()
X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
X = np.column_stack([X, np.array([CORRIDORS.index(c) for c, _, _ in index], float)])
names = names + ["corridor_id"]
dates = np.array([d for _, _, d in index], dtype=object)
Y = build_targets(s, index)
RATE = reference_rate(
    BASELINES["ТЗ: уровень (нижний дециль)"](X[:, :-1], names[:-1]), dates, 2021)

# Выбор ДО теста: для каждого горизонта — модель с лучшим AUC на внутренней
# валидации периода разработки, отдельно на всех признаках и на отобранных.
CASES: list[tuple[int, str, bool, bool]] = []
for _h in (3, 5, 10):
    _y = Y[f"fav_h{_h}"]
    _cols, _, _ = select_features(X, _y, dates, names, 2021)
    _n_all, _r_all = select_model(X, _y, dates, 2021)
    _n_sel, _r_sel = select_model(X, _y, dates, 2021, cols=list(_cols))
    if dict(_r_all)[_n_all] >= dict(_r_sel)[_n_sel]:
        CASES.append((_h, _n_all, False, True))
    else:
        CASES.append((_h, _n_sel, True, True))
# справочно — то, что выигрывало на тесте (результатом не является)
CASES += [
    (3, "случайный лес", True, False),
    (5, "CatBoost", False, False),
    (10, "логистическая регрессия", False, False),
]

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
        cols, _, _ = select_features(X, y, dates, names, 2021)
    oos = np.zeros(len(y), bool)
    sc = np.full(len(y), np.nan)
    f = np.zeros(len(y), bool)
    for tr_i, te_i, _ in walk_forward_folds(dates, 2021, h):
        assert_no_overlap(dates, tr_i, te_i, h)
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
    lf, _, _ = lift(f, y)
    tag = "отбор" if use_sel else "все"
    mark = "выбрано ДО теста" if chosen_before else "СПРАВОЧНО: выигрывало на тесте"
    print(f"\n  {mname} [{tag}], h={h}  |  lift {lf:.2f}, "
          f"достижимая {np.nanmean(np.where(f, fwd, np.nan)):+.0f} бп   ({mark})")
    yrs = sorted({d.year for d in dates[oos]})
    print("   " + "".join(f"{yy:>11}" for yy in yrs))
    r1 = r2 = ""
    pos = 0
    for yy in yrs:
        msk = np.array([d.year == yy for d in dates]) & f & ~np.isnan(fwd)
        if msk.sum() > 15:
            val = np.nanmean(fwd[msk])
            r1 += f"{val:>+8.0f}бп"
            pos += val > 0
        else:
            r1 += f"{'—':>11}"
        r2 += f"{'n=' + str(int(msk.sum())):>11}"
    print("   " + r1)
    print("   " + r2)
    print(f"   лет с положительным результатом: {pos} из {len(yrs)}")
