"""Основной эксперимент: признаки без заглядывания -> walk-forward -> uplift к ТЗ.

Запуск:  .venv/bin/python run_experiment.py
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import bootstrap_ci, lift, mean_benefit, reference_rate, train_cutoff
from ml.features import build_matrix
from ml.leakage import check_detector_works, check_no_lookahead
from ml.models import make_classifiers
from ml.selection import select_model
from ml.targets import HORIZONS, benefit_forward_only, build_targets
from ml.validation import assert_no_overlap, walk_forward_folds

FIRST_TEST_YEAR = 2021


def gate_leakage(series) -> None:
    """Ворота: без доказанного отсутствия утечки эксперимент не запускается."""
    print("=" * 100)
    print("ВОРОТА 1. ЗАГЛЯДЫВАНИЕ В БУДУЩЕЕ")
    print("=" * 100)
    cut = dt.date(2023, 6, 30)
    if not check_detector_works(series, cut):
        sys.exit("ОСТАНОВ: проверка не ловит подставную утечку, её вердикт ничего не значит")
    print("  детектор ловит подставную утечку (сдвиг среза на 5 дней) ...... ОК")
    clean, leaked, n = check_no_lookahead(series, cut)
    if not clean:
        sys.exit(f"ОСТАНОВ: признаки смотрят в будущее: {leaked}")
    print(f"  порча будущего не меняет признаки прошлого ({n} строк) ......... ОК")
    clean2, leaked2, n2 = check_no_lookahead(series, dt.date(2021, 3, 15), factor=0.2)
    if not clean2:
        sys.exit(f"ОСТАНОВ: признаки смотрят в будущее: {leaked2}")
    print(f"  то же на второй дате среза ({n2} строк) ........................ ОК\n")


def main() -> None:
    series = load()
    gate_leakage(series)

    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    # коридор как признак: модель одна на всех, но знает, в каком коридоре стоит
    corr_ids = np.array([CORRIDORS.index(c) for c, _, _ in index], dtype=float)
    X = np.column_stack([X, corr_ids])
    names = names + ["corridor_id"]
    dates = np.array([d for _, _, d in index], dtype=object)
    corridors = np.array([c for c, _, _ in index], dtype=object)
    Y = build_targets(series, index)
    fwd_all = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        b = benefit_forward_only(series[c].values, i, 5)
        if b is not None:
            fwd_all[r] = b
    span_years = (max(dates) - min(dates)).days / 365.25
    wall = dt.date(FIRST_TEST_YEAR, 1, 1)
    pre = np.array([d < wall for d in dates])

    print("=" * 100)
    print("ДАННЫЕ")
    print("=" * 100)
    print(f"  строк {X.shape[0]}, признаков {X.shape[1]}, коридоров {len(CORRIDORS)}")
    print(f"  период {min(dates)} .. {max(dates)} ({span_years:.1f} года)")
    print(f"  обучение расширяющимся окном, тест — календарный год с {FIRST_TEST_YEAR}\n")

    for target_name, target_label in (("fav", "СЕЙЧАС ВЫГОДНО"), ("close", "ОКНО ЗАКРЫВАЕТСЯ")):
        for h in HORIZONS:
            y = Y[f"{target_name}_h{h}"]
            ben = Y[f"benefit_h{h}"]
            fwd = np.full(len(index), np.nan)
            for r, (c, i, _) in enumerate(index):
                b = benefit_forward_only(series[c].values, i, h)
                if b is not None:
                    fwd[r] = b
            folds = walk_forward_folds(dates, FIRST_TEST_YEAR, h)
            if not folds:
                continue

            # Целевая частота — частота правила ТЗ на периоде РАЗРАБОТКИ.
            # Считать её на тесте нельзя: это уже подгонка рабочей точки под тест.
            ref_rate = reference_rate(
                BASELINES["ТЗ: уровень (нижний дециль)"](X[:, :-1], names[:-1]),
                dates, FIRST_TEST_YEAR)

            oos = np.zeros(len(y), dtype=bool)
            scores: dict[str, np.ndarray] = {m: np.full(len(y), np.nan) for m in make_classifiers()}
            fires: dict[str, np.ndarray] = {m: np.zeros(len(y), bool) for m in make_classifiers()}
            for train_idx, test_idx, _year in folds:
                assert_no_overlap(dates, train_idx, test_idx, h)
                tr = train_idx[~np.isnan(y[train_idx])]
                te = test_idx[~np.isnan(y[test_idx])]
                if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
                    continue
                oos[te] = True
                for mname, model in make_classifiers().items():
                    model.fit(X[tr], y[tr])
                    scores[mname][te] = model.predict_proba(X[te])[:, 1]
                    cut = train_cutoff(model.predict_proba(X[tr])[:, 1], ref_rate)
                    fires[mname][te] = scores[mname][te] >= cut

            if oos.sum() < 200:
                continue
            base_rate = float(np.nanmean(y[oos]))

            print("=" * 104)
            print(f"ЦЕЛЬ: {target_label}   горизонт h = {h}   out-of-sample строк {int(oos.sum())}")
            print(f"базовая ставка (случайный день) = {base_rate*100:.1f} %")
            print("=" * 104)
            print(
                f"{'правило / модель':<32}{'частота':>9}{'сигн/нед':>10}{'lift':>7}"
                f"{'выгода ±h':>12}{'ДОСТИЖИМАЯ':>13}{'95% ДИ':>18}"
            )

            def stats(fired: np.ndarray) -> tuple[float, float, float, float, tuple[float, float]]:
                lf, _, n = lift(fired, y)
                pw = n / len(CORRIDORS) / (span_years * 52 * (oos.sum() / len(y)))
                sym = mean_benefit(fired, ben)
                fw = float(np.nanmean(np.where(fired, fwd, np.nan)))
                ci = bootstrap_ci(fwd[fired & ~np.isnan(fwd)])
                return lf, pw, sym, fw, ci

            tz_rows, model_rows = [], []
            for bname, bfn in BASELINES.items():
                fired = bfn(X[:, :-1], names[:-1]).astype(bool) & oos
                rate = fired.sum() / max(int(oos.sum()), 1)
                lf, pw, sym, fw, ci = stats(fired)
                row = (bname, rate, pw, lf, sym, fw, ci)
                (model_rows if False else tz_rows).append(row)
                print(
                    f"    {bname:<28}{rate*100:>8.1f}%{pw:>10.2f}{lf:>7.2f}"
                    f"{sym:>+11.0f}бп{fw:>+12.0f}бп{f'[{ci[0]:+.0f}; {ci[1]:+.0f}]':>18}"
                )

            for mname in scores:
                fired = fires[mname] & oos
                rate = fired.sum() / max(int(oos.sum()), 1)
                lf, pw, sym, fw, ci = stats(fired)
                model_rows.append((mname, rate, pw, lf, sym, fw, ci))
                print(
                    f"  * {mname:<28}{rate*100:>8.1f}%{pw:>10.2f}{lf:>7.2f}"
                    f"{sym:>+11.0f}бп{fw:>+12.0f}бп{f'[{ci[0]:+.0f}; {ci[1]:+.0f}]':>18}"
                )

            # UPLIFT считается ТОЛЬКО против индикаторов ТЗ; контрпример исключён.
            # Модель фиксируется ДО теста — по внутренней валидации периода разработки.
            # Брать лучшую на тесте нельзя: это выбор рабочей точки по тесту, только
            # на уровне модели, и он завышает результат тем сильнее, чем больше перебор.
            tz_only = [r for r in tz_rows if r[0].startswith("ТЗ:")]
            best_tz_lift = max(tz_only, key=lambda r: r[3])
            best_tz_fwd = max(tz_only, key=lambda r: r[5])
            chosen, sel_rep = select_model(X, y, dates, FIRST_TEST_YEAR)
            md = dict((r[0], r) for r in model_rows)[chosen]
            print(f"\n  модель выбрана ДО теста (AUC на внутренней валидации): {chosen}"
                  f"  [{', '.join(f'{n} {a:.3f}' for n, a in sel_rep)}]")
            print(
                f"  UPLIFT по lift:              {md[3]:.2f} ({chosen})"
                f"  против {best_tz_lift[3]:.2f} ({best_tz_lift[0]})"
                f"  = {md[3]-best_tz_lift[3]:+.2f}"
            )
            print(
                f"  UPLIFT по достижимой выгоде: {md[5]:+.0f} бп ({chosen})"
                f"  против {best_tz_fwd[5]:+.0f} бп ({best_tz_fwd[0]})"
                f"  = {md[5]-best_tz_fwd[5]:+.0f} бп"
            )
            mx_l = max(model_rows, key=lambda r: r[3])
            mx_f = max(model_rows, key=lambda r: r[5])
            print(f"  справочно, лучшая НА ТЕСТЕ (как результат не годится): "
                  f"lift {mx_l[3]:.2f} ({mx_l[0]}), выгода {mx_f[5]:+.0f} бп ({mx_f[0]})")
            print()


if __name__ == "__main__":
    main()
