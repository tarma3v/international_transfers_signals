"""Бустинги (CatBoost, XGBoost) + честный отбор признаков против индикаторов ТЗ.

Отбор признаков идёт ТОЛЬКО по данным до первого тестового года — иначе тест
участвовал бы в решении, какие признаки брать, и результат был бы завышен.
"""
from __future__ import annotations

import sys

import numpy as np

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import REFERENCE_RULE, UPLIFT_COMPARATOR, bootstrap_ci, lift, rate_per_week, reference_rate, train_cutoff
from ml.features import build_matrix
from ml.leakage import check_detector_works, check_no_lookahead
from ml.models import make_classifiers
from ml.selection import select_features, select_model
from ml.targets import benefit_forward_only, build_targets
from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds

import datetime as dt

FIRST_TEST_YEAR = 2021
HORIZONS = (3, 5, 10)


def gate(series) -> None:
    if not check_detector_works(series, dt.date(2023, 6, 30)):
        sys.exit("ОСТАНОВ: проверка не ловит подставную утечку")
    for cutd, fac in ((dt.date(2023, 6, 30), 3.0), (dt.date(2021, 3, 15), 0.2)):
        clean, leaked, _ = check_no_lookahead(series, cutd, fac)
        if not clean:
            sys.exit(f"ОСТАНОВ: признаки смотрят в будущее: {leaked}")
    print("ВОРОТА: заглядывания в будущее нет (детектор проверен, 2 среза чисты)\n")


def main() -> None:
    s = load()
    gate(s)
    X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    Y = build_targets(s, index)
    span = (max(dates) - min(dates)).days / 365.25
    RATE = reference_rate(
        BASELINES[REFERENCE_RULE](X, names), dates, FIRST_TEST_YEAR)

    for h in HORIZONS:
        y = Y[f"fav_h{h}"]
        fwd = np.full(len(index), np.nan)
        for r, (c, i, _) in enumerate(index):
            b = benefit_forward_only(s[c].values, i, h)
            if b is not None:
                fwd[r] = b

        cols, k, rep = select_features(X, y, dates, names, FIRST_TEST_YEAR, horizon=h, reach=target_reach_dates(index, s, h))
        reach = target_reach_dates(index, s, h)
        folds = walk_forward_folds(dates, FIRST_TEST_YEAR, h, reach=reach)
        oos = np.zeros(len(y), bool)
        sc: dict[str, np.ndarray] = {}
        fires: dict[str, np.ndarray] = {}
        for mname in make_classifiers():
            for tag in ("все", "отбор"):
                sc[f"{mname} [{tag}]"] = np.full(len(y), np.nan)
                fires[f"{mname} [{tag}]"] = np.zeros(len(y), bool)

        for tr_i, te_i, _ in folds:
            assert_no_overlap(dates, tr_i, te_i, h, index=index, series=s)
            tr = tr_i[~np.isnan(y[tr_i])]
            te = te_i[~np.isnan(y[te_i])]
            if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
                continue
            oos[te] = True
            # Порог берётся по обучающим оценкам фолда, а не по тестовым.
            for mname, mdl in make_classifiers().items():
                mdl.fit(X[tr], y[tr])
                sc[f"{mname} [все]"][te] = mdl.predict_proba(X[te])[:, 1]
                cut = train_cutoff(mdl.predict_proba(X[tr])[:, 1], RATE)
                fires[f"{mname} [все]"][te] = sc[f"{mname} [все]"][te] >= cut
            for mname, mdl in make_classifiers().items():
                mdl.fit(X[tr][:, cols], y[tr])
                sc[f"{mname} [отбор]"][te] = mdl.predict_proba(X[te][:, cols])[:, 1]
                cut = train_cutoff(mdl.predict_proba(X[tr][:, cols])[:, 1], RATE)
                fires[f"{mname} [отбор]"][te] = sc[f"{mname} [отбор]"][te] >= cut

        base = float(np.nanmean(y[oos]))
        print("=" * 104)
        print(f"ЦЕЛЬ «СЕЙЧАС ВЫГОДНО», h = {h}   |   OOS строк {int(oos.sum())}, базовая ставка {base*100:.1f} %")
        print(f"отбор признаков: K = {k} из {len(names)}, выбрано на данных до {FIRST_TEST_YEAR}")
        print("=" * 104)
        print(f"{'правило / модель':<34}{'частота':>9}{'сигн/нед':>10}{'lift':>7}{'ДОСТИЖИМАЯ':>13}{'95% ДИ':>18}")

        def show(nm: str, fired: np.ndarray, star: bool) -> tuple[float, float]:
            lf, _, n = lift(fired, y, scope=oos)
            rate = fired.sum() / max(int(oos.sum()), 1)
            pw = rate_per_week(n, len(CORRIDORS), dates, oos)
            fw = float(np.nanmean(np.where(fired, fwd, np.nan)))
            sel = fired & ~np.isnan(fwd)
            lo, hi = bootstrap_ci(fwd[sel], dates=dates[sel])
            print(f"{'  * ' if star else '    '}{nm:<30}{rate*100:>8.1f}%{pw:>10.2f}{lf:>7.2f}"
                  f"{fw:>+12.0f}бп{f'[{lo:+.0f}; {hi:+.0f}]':>18}")
            return lf, fw

        tz = []
        for bn, bf in BASELINES.items():
            f = bf(X, names).astype(bool) & oos
            lf, fw = show(bn, f, False)
            if bn.startswith("ТЗ:"):
                tz.append((bn, lf, fw))
        rows = []
        for nm, s_ in sc.items():
            lf, fw = show(nm, fires[nm] & oos, True)
            rows.append((nm, lf, fw))

        cmp_row = dict((r[0], r) for r in tz)[UPLIFT_COMPARATOR]
        bt_l = max(tz, key=lambda r: r[1])
        bt_f = max(tz, key=lambda r: r[2])
        # Конфигурация фиксируется ДО теста: модель — по внутренней валидации
        # периода разработки, набор признаков — тем же отбором. Максимум по тесту
        # результатом быть не может.
        best_all, rep_all = select_model(X, y, dates, FIRST_TEST_YEAR, horizon=h, reach=target_reach_dates(index, s, h))
        best_sel, rep_sel = select_model(X, y, dates, FIRST_TEST_YEAR, cols=list(cols), horizon=h, reach=target_reach_dates(index, s, h))
        auc_all = dict(rep_all)[best_all]
        auc_sel = dict(rep_sel)[best_sel]
        chosen = f"{best_all} [все]" if auc_all >= auc_sel else f"{best_sel} [отбор]"
        md = dict((r[0], r) for r in rows)[chosen]
        print(f"\n  конфигурация выбрана ДО теста: {chosen}  (AUC на внутренней валидации "
              f"{max(auc_all, auc_sel):.3f})")
        # Полный рейтинг внутренней валидации: без него «выбрано до теста» —
        # утверждение на слово. С ним видно и победителя, и на сколько он
        # опередил следующего, и что тестовые числа в решении не участвовали.
        ranking = ([(f"{nm} [все]", a) for nm, a in rep_all]
                   + [(f"{nm} [отбор]", a) for nm, a in rep_sel])
        ranking.sort(key=lambda r: -r[1])
        print("  рейтинг внутренней валидации (тест в нём не участвует):")
        for nm, a in ranking:
            mark = "  <- выбрана" if nm == chosen else ""
            print(f"      {nm:<38}AUC {a:.3f}{mark}")
        print(f"  компаратор назначен ДО теста: {UPLIFT_COMPARATOR}")
        print(f"  UPLIFT lift:               {md[1]:.2f} против {cmp_row[1]:.2f} = {md[1]-cmp_row[1]:+.2f}")
        print(f"  UPLIFT достижимая выгода:  {md[2]:+.0f} бп против {cmp_row[2]:+.0f} бп = {md[2]-cmp_row[2]:+.0f} бп")
        print(f"  справочно, разность к максимуму по тесту (lift {bt_l[0]}, выгода {bt_f[0]}): "
              f"{md[1]-bt_l[1]:+.2f} / {md[2]-bt_f[2]:+.0f} бп — max-статистика, не парная оценка")
        mx_l = max(rows, key=lambda r: r[1]); mx_f = max(rows, key=lambda r: r[2])
        print(f"  справочно, лучшая НА ТЕСТЕ (результатом не является): "
              f"lift {mx_l[1]:.2f} ({mx_l[0]}), выгода {mx_f[2]:+.0f} бп ({mx_f[0]})\n")


if __name__ == "__main__":
    main()
