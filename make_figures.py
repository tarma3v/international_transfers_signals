"""Графики для презентации и дополнительных материалов."""
from __future__ import annotations

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

from ml.data import CORRIDORS, REFERENCE, load
from ml.features import WARMUP, build_matrix
from ml.targets import benefit_backward_only, benefit_forward_only

rcParams["font.family"] = ["DejaVu Sans"]
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["figure.dpi"] = 140

INK = "#1a2530"
ACCENT = "#9C5B12"
POS = "#16674F"
NEG = "#9E3226"
GREY = "#8a94a0"
OUT = "submission/figures"


def save(fig, name: str) -> None:
    """Сохранить фигуру в PNG и PDF.

    PNG идёт в презентацию и в markdown, PDF — в комплект материалов: он
    векторный, поэтому в нём читается ось и подпись при любом увеличении,
    а растр на проекторе рассыпается. Два формата из одного вызова, чтобы
    они не разъезжались.
    """
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")

s = load()
H = 5


def fig1_corridor_with_signals() -> None:
    """График коридора с отметками срабатываний — кейс требует его на защите."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    c = "TJS"
    dates = list(s[c].dates)
    vals = list(s[c].values)
    ax.plot(dates, vals, color=INK, lw=1.0, label="Курс ЦБ, ₽ за 1 сомони")

    lo_idx, hi_idx = [], []
    for i in range(90, len(vals)):
        # те же 90 точек, что берёт _position_in_range(past, 90)
        w = vals[i - 89 : i + 1]
        rng = max(w) - min(w)
        if rng <= 0:
            continue
        p = (vals[i] - min(w)) / rng
        if p <= 0.10:
            lo_idx.append(i)
        elif p >= 0.95:
            hi_idx.append(i)
    ax.scatter([dates[i] for i in lo_idx], [vals[i] for i in lo_idx], s=9, color=POS,
               zorder=3, label=f"Зелёный: нижние 10 % квартального диапазона ({len(lo_idx)} дней)")
    ax.scatter([dates[i] for i in hi_idx], [vals[i] for i in hi_idx], s=9, color=NEG,
               zorder=3, label=f"Красный: верхние 5 % ({len(hi_idx)} дней)")
    ax.set_title("Коридор Россия → Таджикистан: когда валюта дёшева и когда дорога",
                 fontsize=12, color=INK, loc="left", pad=12)
    ax.set_ylabel("₽ за 1 сомони", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    save(fig, "01-koridor-s-signalami")
    plt.close(fig)


def fig2_traffic_light() -> None:
    """Светофор: две метрики дают ПРОТИВОПОЛОЖНЫЕ цвета.

    Симметричная выгода ±h для светофора не годится: половина её уже случилась
    и клиенту недоступна. По ней дешёвые дни выглядят выгодными, а дорогие —
    вредными; по достижимой половине всё наоборот. Светофор, откалиброванный
    по ±h, отправлял бы клиента переводить ровно в те дни, когда это хуже всего.
    Поэтому рисуются обе метрики рядом.
    """
    BK = [(0, 10, "0–10 %\nдёшево"), (10, 30, "10–30 %"), (30, 70, "30–70 %\nнейтрально"),
          (70, 90, "70–90 %"), (90, 101, "90–100 %\nдорого")]
    agg = {b[2]: [] for b in BK}
    fwd_b = {b[2]: [] for b in BK}
    hits = {b[2]: [] for b in BK}
    for c in CORRIDORS:
        v = list(s[c].values)
        # тот же прогрев, что и везде: иначе бакеты в графике и в отчёте
        # посчитаны на разных выборках и цифры расходятся
        for i in range(WARMUP, len(v) - H):
            # ровно те же 90 точек, что берёт _position_in_range(past, 90):
            # окно включает сегодняшний день, поэтому срез начинается с i-89
            w = v[i - 89 : i + 1]
            rng = max(w) - min(w)
            if rng <= 0:
                continue
            p = (v[i] - min(w)) / rng * 100
            ref = float(np.mean(v[i - H : i + H + 1]))
            ben = -(v[i] - ref) / ref * 10000
            fw = benefit_forward_only(s[c].values, i, H)
            for a, b, nm in BK:
                if a <= p < b:
                    agg[nm].append(ben)
                    if fw is not None:
                        fwd_b[nm].append(fw)
                    hits[nm].append(1.0 if v[i] <= min(v[i + 1 : i + H + 1]) else 0.0)
                    break
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [b[2] for b in BK]
    x = np.arange(5)
    sym = [np.mean(agg[l]) for l in labels]
    fw = [np.mean(fwd_b[l]) for l in labels]
    ax1.bar(x - 0.2, sym, 0.4, label="выгода ±h (в метрике ТЗ)", color=GREY)
    ax1.bar(x + 0.2, fw, 0.4, label="ДОСТИЖИМАЯ половина", color=ACCENT)
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Выгода, б. п.", fontsize=9)
    ax1.set_title("Две метрики дают противоположный светофор", fontsize=11, color=INK, loc="left")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(axis="y", alpha=0.15)
    for xi, a, b in zip(x, sym, fw):
        ax1.annotate(f"{a:+.0f}", (xi - 0.2, a), textcoords="offset points",
                     xytext=(0, 4 if a >= 0 else -12), ha="center", fontsize=8, color=GREY)
        ax1.annotate(f"{b:+.0f}", (xi + 0.2, b), textcoords="offset points",
                     xytext=(0, 4 if b >= 0 else -12), ha="center", fontsize=8,
                     color=ACCENT, weight="bold")
    hr = [np.mean(hits[l]) * 100 for l in labels]
    ax2.bar(x, hr, color=ACCENT)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Попадание по метрике ТЗ, %", fontsize=9)
    ax2.set_title("Попадание растёт к дорогим дням", fontsize=11, color=INK, loc="left")
    ax2.grid(axis="y", alpha=0.15)
    print(f"  fig2: ±h дёшево {sym[0]:+.0f} / дорого {sym[-1]:+.0f}; "
          f"достижимая дёшево {fw[0]:+.0f} / дорого {fw[-1]:+.0f}")
    fig.tight_layout()
    save(fig, "02-konflikt-metrik")
    plt.close(fig)


def fig3_decomposition() -> None:
    """Разложение выгоды. Всё считается на ОДНИХ И ТЕХ ЖЕ out-of-sample строках,
    модель — через walk-forward, без захардкоженных чисел."""
    from ml.evaluate import train_cutoff
    from ml.models import make_classifiers
    from ml.selection import select_model
    from ml.targets import build_targets
    from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds

    X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
    Xm = np.column_stack([X, np.array([CORRIDORS.index(c) for c, _, _ in index], float)])
    dates = np.array([d for _, _, d in index], dtype=object)
    y = build_targets(s, index)[f"fav_h{H}"]

    fwd = np.full(len(index), np.nan)
    bwd = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        a = benefit_forward_only(s[c].values, i, H)
        b = benefit_backward_only(s[c].values, i, H)
        if a is not None:
            fwd[r] = a
        if b is not None:
            bwd[r] = b

    # Модель фиксируется ДО теста — иначе на график попадёт победитель теста.
    chosen, _ = select_model(Xm, y, dates, 2021, horizon=H, reach=target_reach_dates(index, s, H))
    # Целевая частота — та же, что в run_experiment.py: доля срабатываний правила
    # «нижний дециль» на периоде РАЗРАБОТКИ. Иначе числа в отчёте и на графике разойдутся.
    from ml.baselines import BASELINES
    from ml.evaluate import REFERENCE_RULE, reference_rate

    ref_rate = reference_rate(BASELINES[REFERENCE_RULE](X, names), dates, 2021)
    oos = np.zeros(len(y), bool)
    sc = np.full(len(y), np.nan)
    model_mask = np.zeros(len(y), bool)
    reach = target_reach_dates(index, s, H)
    for tr_i, te_i, _ in walk_forward_folds(dates, 2021, H, reach=reach):
        assert_no_overlap(dates, tr_i, te_i, H, index=index, series=s)
        tr = tr_i[~np.isnan(y[tr_i])]
        te = te_i[~np.isnan(y[te_i])]
        if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
            continue
        oos[te] = True
        m = make_classifiers()[chosen]
        m.fit(Xm[tr], y[tr])
        sc[te] = m.predict_proba(Xm[te])[:, 1]
        # порог — из обучения фолда
        model_mask[te] = sc[te] >= train_cutoff(m.predict_proba(Xm[tr])[:, 1], ref_rate)

    pct = X[:, names.index("pct_range_90")]
    dn = X[:, names.index("streak_dn")]
    model_mask &= oos

    rules = {
        "ТЗ: моментум\n(падение 3 дня)": (dn >= 3) & oos,
        "ТЗ: уровень\n(нижний дециль)": (pct <= 10) & oos,
        f"Модель\n({chosen})": model_mask,
        "Простое правило\n(верх диапазона)": (pct >= 95) & oos,
    }
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels, fvals, bvals = [], [], []
    for nm, mask in rules.items():
        labels.append(nm)
        fvals.append(float(np.nanmean(np.where(mask, fwd, np.nan))))
        bvals.append(float(np.nanmean(np.where(mask, bwd, np.nan))))
    x = np.arange(len(labels))
    ax.bar(x - 0.19, bvals, 0.38, label="Недостижимая половина (уже случилось)", color=GREY)
    ax.bar(x + 0.19, fvals, 0.38,
           color=[POS if v > 0 else NEG for v in fvals],
           label="ДОСТИЖИМАЯ половина (клиент может забрать)")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Выгода, б. п.", fontsize=9)
    ax.set_title("Выгода по половинам, горизонт h = 5", fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.15)
    lo = min(min(fvals), 0) - 30
    ax.set_ylim(lo, max(bvals) * 1.28)
    for xi, v in zip(x + 0.19, fvals):
        ax.annotate(f"{v:+.0f}", (xi, v), textcoords="offset points",
                    xytext=(0, 8 if v > 0 else -16), ha="center", fontsize=10, weight="bold",
                    color=POS if v > 0 else NEG)
    ax.annotate("out-of-sample 2021–2026, порог модели зафиксирован на обучении",
                (0, 0), xycoords="axes fraction", xytext=(0, -46), textcoords="offset points",
                fontsize=8, color=GREY)
    fig.tight_layout()
    save(fig, "03-razlozhenie-vygody")
    plt.close(fig)
    print(f"  fig3: моментум {fvals[0]:+.0f}, уровень {fvals[1]:+.0f}, {chosen} {fvals[2]:+.0f} бп")


def fig4_stability() -> None:
    """Устойчивость по годам: конфигурация, выбранная ДО теста, против победителя теста.

    Ничего не захардкожено — обе серии считаются здесь же, walk-forward,
    порог срабатывания из обучения каждого фолда.
    """
    from ml.evaluate import train_cutoff
    from ml.models import make_classifiers
    from ml.selection import select_features, select_model
    from ml.targets import build_targets
    from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds

    X, names, index = build_matrix(s, CORRIDORS, REFERENCE)
    Xm = np.column_stack([X, np.array([CORRIDORS.index(c) for c, _, _ in index], float)])
    dates = np.array([d for _, _, d in index], dtype=object)
    y = build_targets(s, index)[f"fav_h{H}"]
    fwd = np.full(len(index), np.nan)
    for r, (c, i, _) in enumerate(index):
        b = benefit_forward_only(s[c].values, i, H)
        if b is not None:
            fwd[r] = b

    from ml.baselines import BASELINES
    from ml.evaluate import REFERENCE_RULE, reference_rate

    ref4 = reference_rate(BASELINES[REFERENCE_RULE](X, names), dates, 2021)
    cols, _, _ = select_features(Xm, y, dates, names + ["corridor_id"], 2021, horizon=H, reach=target_reach_dates(index, s, H))
    n_all, r_all = select_model(Xm, y, dates, 2021, horizon=H, reach=target_reach_dates(index, s, H))
    n_sel, r_sel = select_model(Xm, y, dates, 2021, cols=list(cols), horizon=H, reach=target_reach_dates(index, s, H))
    honest = (n_all, None) if dict(r_all)[n_all] >= dict(r_sel)[n_sel] else (n_sel, list(cols))

    configs: list[tuple[str, str, list[int] | None]] = []
    for mname in make_classifiers():
        configs.append((f"{mname} [все]", mname, None))
        configs.append((f"{mname} [отбор]", mname, list(cols)))

    oos = np.zeros(len(y), bool)
    fires = {lbl: np.zeros(len(y), bool) for lbl, _, _ in configs}
    reach = target_reach_dates(index, s, H)
    for tr_i, te_i, _ in walk_forward_folds(dates, 2021, H, reach=reach):
        assert_no_overlap(dates, tr_i, te_i, H, index=index, series=s)
        tr = tr_i[~np.isnan(y[tr_i])]
        te = te_i[~np.isnan(y[te_i])]
        if len(tr) < 400 or len(te) < 30 or len(np.unique(y[tr])) < 2:
            continue
        oos[te] = True
        for lbl, mname, cc in configs:
            use = slice(None) if cc is None else cc
            m = make_classifiers()[mname]
            m.fit(Xm[tr][:, use], y[tr])
            cut = train_cutoff(m.predict_proba(Xm[tr][:, use])[:, 1], ref4)
            fires[lbl][te] = m.predict_proba(Xm[te][:, use])[:, 1] >= cut

    honest_lbl = f"{honest[0]} [{'все' if honest[1] is None else 'отбор'}]"
    winner_lbl = max(configs, key=lambda cf: np.nanmean(
        np.where(fires[cf[0]] & oos, fwd, np.nan)))[0]

    yrs = sorted({d.year for d in dates[oos]})
    year_of = np.array([d.year for d in dates])

    def by_year(lbl: str) -> tuple[list[float], list[int]]:
        vals, cnts = [], []
        for yy in yrs:
            msk = (year_of == yy) & fires[lbl] & oos & ~np.isnan(fwd)
            cnts.append(int(msk.sum()))
            vals.append(float(np.nanmean(fwd[msk])) if msk.sum() > 15 else 0.0)
        return vals, cnts

    hv, hn = by_year(honest_lbl)
    wv, wn = by_year(winner_lbl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(yrs))
    ax1.bar(x - 0.19, hv, 0.38, label=f"выбрана ДО теста: {honest_lbl}", color=ACCENT)
    ax1.bar(x + 0.19, wv, 0.38, label=f"победитель теста: {winner_lbl}", color=POS)
    ax1.axhline(0, color=INK, lw=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(yrs, fontsize=9)
    ax1.set_ylabel("Достижимая выгода, б. п.", fontsize=9)
    ax1.set_title("Результат по годам", fontsize=11, color=INK, loc="left")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(axis="y", alpha=0.15)
    ax2.plot(x, hn, "o-", color=ACCENT, label=honest_lbl)
    ax2.plot(x, wn, "o-", color=POS, label=winner_lbl)
    ax2.set_xticks(x)
    ax2.set_xticklabels(yrs, fontsize=9)
    ax2.set_ylabel("Сработало дней за год", fontsize=9)
    ax2.set_title("Сколько раз модель вообще подала сигнал", fontsize=11, color=INK, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    save(fig, "04-ustoychivost")
    plt.close(fig)
    print(f"  fig4: {honest_lbl} {np.mean([v for v in hv if v]):+.0f} бп/год, "
          f"победитель теста {winner_lbl} {np.mean([v for v in wv if v]):+.0f} бп/год")


def _parse_two_models() -> tuple[list[float], list[int], list[int], list[int], int]:
    """Цифры берём из свежего прогона, а не из памяти.

    Захардкоженное число в графике расходится с кодом молча: прогон меняется,
    картинка остаётся прежней, и заметить это можно только сверкой глазами.
    Разбор намеренно падает с исключением, если строка не найдена."""
    import re

    txt = Path("results/two_models_output.txt").read_text(encoding="utf-8")
    lifts, gains, los, his = [], [], [], []
    # В таблице метрики кейса после lift идёт колонка «полоса ТЗ» — её и якорим,
    # иначе регулярка молча захватит не тот столбец.
    # Якорим на колонку «полоса ТЗ» — она есть только в таблице метрики кейса,
    # поэтому подпись строки можно менять, не ломая разбор.
    tail = r"\s+([\d.]+)\s+(?:НИЖЕ|ВЫШЕ|в полосе)\s*$"
    for pat in (r"простое правило: верх диапазона.*?" + tail,
                r"МОДЕЛЬ A \(.*?" + tail,
                r"МОДЕЛЬ B \(.*?" + tail):
        m = re.search(pat, txt, re.M)
        if m is None:
            raise RuntimeError(f"не нашёл строку lift: {pat}")
        lifts.append(float(m.group(1)))
    mo = re.search(r"Потолок \(оракул, знает будущее\): \+(\d+) бп", txt)
    if mo is None:
        raise RuntimeError("не нашёл потолок оракула — он не должен быть захардкожен")
    oracle = int(mo.group(1))
    for pat in (r"Правило: верхние 5 % диапазона\s*([-+]\d+)бп\s*\[([-+]\d+); ([-+]\d+)\]",
                r"МОДЕЛЬ A \(метрика кейса\)\s*([-+]\d+)бп\s*\[([-+]\d+); ([-+]\d+)\]",
                r"МОДЕЛЬ B \(порог: предсказание > 0\)\s*([-+]\d+)бп\s*\[([-+]\d+); ([-+]\d+)\]"):
        m = re.search(pat, txt)
        if m is None:
            raise RuntimeError(f"не нашёл строку выгоды: {pat}")
        gains.append(int(m.group(1))); los.append(int(m.group(2))); his.append(int(m.group(3)))
    return lifts, gains, los, his, oracle


def fig5_two_models() -> None:
    """Две модели под две метрики. Цифры читаются из results/two_models_output.txt."""
    lifts, gains, los, his, oracle = _parse_two_models()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    # «МОДЕЛЬ B» в двух панелях — РАЗНЫЕ политики: слева порог берётся из обучения,
    # справа сигналом считается любое положительное предсказание. Одна подпись на
    # обе панели склеивала бы их в одну сущность, поэтому подписи разные.
    # У «МОДЕЛИ A» строка-источник помечена «набор признаков с теста — справочно».
    # Без этой пометки столбец читается как честный результат, а он им не является.
    # Подписи в три строки: в одну строку они наезжают на соседний столбец.
    labels_lift = ["Правило\n«верхние 5 %»",
                   "МОДЕЛЬ A\nпризнаки с теста,\nсправочно",
                   "МОДЕЛЬ B\nпорог\nиз обучения"]
    labels_gain = ["Правило\n«верхние 5 %»",
                   "МОДЕЛЬ A\nпризнаки с теста,\nсправочно",
                   "МОДЕЛЬ B\nпорог:\nпредсказание > 0"]
    x = np.arange(3)
    ax1.bar(x, lifts, 0.55, color=[GREY, ACCENT, POS])
    ax1.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax1.annotate("уровень случайного дня", (0.015, 1.0), xycoords=("axes fraction", "data"),
                 textcoords="offset points", xytext=(0, 4),
                 fontsize=8, color=INK, ha="left", va="bottom")
    ax1.set_xticks(x); ax1.set_xticklabels(labels_lift, fontsize=8)
    ax1.set_ylim(min(0.95, min(lifts) - 0.05), max(lifts) + 0.12)
    ax1.set_ylabel("lift по метрике кейса", fontsize=9)
    ax1.set_title("Метрика заказчика", fontsize=11, color=INK, loc="left")
    ax1.grid(axis="y", alpha=0.15)
    for xi, v in zip(x, lifts):
        ax1.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=10, weight="bold")

    ax2.bar(x, gains, 0.55, color=[GREY, ACCENT, POS])
    ax2.errorbar(x, gains, yerr=[np.array(gains) - np.array(los), np.array(his) - np.array(gains)],
                 fmt="none", ecolor=INK, capsize=5, lw=1.1)
    ax2.axhline(0, color=INK, lw=0.9)
    ax2.axhline(oracle, color=NEG, lw=1.1, ls="--")
    ax2.annotate(f"потолок оракула +{oracle} бп", (0.015, oracle), xycoords=("axes fraction", "data"),
                 textcoords="offset points", xytext=(0, -6),
                 fontsize=8, color=NEG, ha="left", va="top")
    ax2.set_xticks(x); ax2.set_xticklabels(labels_gain, fontsize=8)
    ax2.set_ylim(min(los) - 15, oracle * 1.15)
    ax2.set_ylabel("бп против дня зарплаты", fontsize=9)
    ax2.set_title("Метрика клиента: что получила семья", fontsize=11, color=INK, loc="left")
    ax2.grid(axis="y", alpha=0.15)
    for xi, v, hi in zip(x, gains, his):
        # Ноль печатаем без знака: округлённый до целых он может прийти как «-0»,
        # и «+0» приписал бы направление, которого в числе нет.
        lab = "0" if v == 0 else f"{v:+.0f}"
        ax2.annotate(lab, (xi, hi), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=10, weight="bold")
    fig.text(0.5, -0.075,
             "«МОДЕЛЬ B» в двух панелях — разные политики: слева порог из обучения, "
             "справа сигнал = любое положительное предсказание.",
             ha="center", fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, "05-dve-modeli")
    plt.close(fig)


if __name__ == "__main__":
    fig1_corridor_with_signals()
    fig2_traffic_light()
    fig3_decomposition()
    fig4_stability()
    fig5_two_models()
    print("готово: 5 графиков (png + pdf) в", OUT)
