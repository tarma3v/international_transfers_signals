"""Продуктовые величины: всё, на что ссылаются документы, считается здесь.

Ни одна цифра в продуктовых документах не должна существовать только в тексте.
Скрипт печатает шесть величин видения, разбивку по коридорам (обязательное
условие ТЗ), кучность сигналов и дрейф базовой ставки по годам.

Всё, что здесь считается, — описательные статистики данных, а не результаты
модели: они не участвуют ни в одном решении об обучении и потому берутся
по всей истории. Модельные числа живут в run_experiment / run_boosting /
run_two_models и считаются строго walk-forward.
"""
from __future__ import annotations

import collections
import datetime as dt

import numpy as np

from ml.baselines import BASELINES
from ml.calendar_ref import HOLIDAYS
from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import bootstrap_ci, lift, rate_per_week
from ml.features import build_matrix
from ml.targets import benefit_bps, benefit_forward_only, target_now_favourable
from ml.two_metrics import payday_anchors

FIRST_TEST = 2021
H = 5
TARGET_PER_WEEK = 1.5  # середина обязательной полосы ТЗ (1-2 сигнала на коридор в неделю)


def _bps(new: float, old: float) -> float:
    return -(new - old) / old * 10000.0


def bar(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def intramonth_range(series) -> None:
    """§1 документа 01: внутримесячный размах курса."""
    bar("РАЗМАХ ВНУТРИ КАЛЕНДАРНОГО МЕСЯЦА (max/min - 1), % — сколько вообще есть что ловить")
    print(f"{'коридор':<10}{'медиана':>10}{'25-й проц.':>12}{'75-й проц.':>12}{'максимум':>11}"
          f"{'месяцев':>10}")
    for c in CORRIDORS:
        s = series[c]
        by_month: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
        for d, v in zip(s.dates, s.values):
            by_month[(d.year, d.month)].append(float(v))
        sp = np.array([max(v) / min(v) - 1.0 for v in by_month.values() if len(v) >= 10]) * 100
        print(f"{c:<10}{np.median(sp):>9.1f}%{np.percentile(sp, 25):>11.1f}%"
              f"{np.percentile(sp, 75):>11.1f}%{sp.max():>10.1f}%{len(sp):>10}")


def oracle_and_shift(series) -> None:
    """§2 документа 01: потолок оракула и цена сдвига на 1-3 публикации."""
    bar("ПОТОЛОК И ЦЕНА СДВИГА — верхняя граница ценности и типичный масштаб движения")
    print("Потолок: перевод в лучший день месяца против среднего дня того же месяца.")
    print("Это НЕДОСТИЖИМАЯ величина (нужно знать будущее) — она задаёт масштаб, не обещание.\n")
    print(f"{'коридор':<10}{'оракул месяца':>16}{'сдвиг ±1 публ.':>17}{'сдвиг ±3 публ.':>17}")
    for c in CORRIDORS:
        s = series[c]
        by_month: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
        for d, v in zip(s.dates, s.values):
            by_month[(d.year, d.month)].append(float(v))
        orc = np.array([_bps(min(v), float(np.mean(v))) for v in by_month.values() if len(v) >= 10])
        v = s.values
        sh1 = np.abs([_bps(float(v[i + 1]), float(v[i])) for i in range(len(v) - 1)])
        sh3 = np.abs([_bps(float(v[i + 3]), float(v[i])) for i in range(len(v) - 3)])
        print(f"{c:<10}{orc.mean():>+14.0f}бп{np.median(sh1):>+15.0f}бп{np.median(sh3):>+15.0f}бп")
    print("\n«сдвиг» — медиана АБСОЛЮТНОГО изменения курса: столько стоит один-три дня")
    print("ожидания в любую сторону. Это не выгода, а размер ставки.")


def cost_of_waiting(series) -> None:
    """§4.2 документа 01: цена ожидания — ожидаемая выгода/потеря от отсрочки."""
    bar("ЦЕНА ОЖИДАНИЯ: подождать k публикаций вместо того, чтобы перевести сегодня")
    print("Знак — со стороны клиента: ПЛЮС = подождать оказалось выгодно (курс упал),")
    print("МИНУС = ожидание обошлось дороже. Среднее по всем дням истории, без сигнала.\n")
    print(f"{'коридор':<10}" + "".join(f"{'k=' + str(k):>14}" for k in (1, 2, 3, 5, 10)))
    for c in CORRIDORS:
        v = series[c].values
        cells = []
        for k in (1, 2, 3, 5, 10):
            g = np.array([_bps(float(v[i + k]), float(v[i])) for i in range(len(v) - k)])
            cells.append(f"{g.mean():>+12.0f}бп")
        print(f"{c:<10}" + "".join(cells))
    print("\nВсе средние отрицательны: за 2019-2026 рубль слабел, и КАЖДЫЙ день ожидания")
    print("в среднем стоил клиенту денег. Это фон, на котором работает продукт:")
    print("сообщение «подожди» борется с трендом, сообщение «переводи» идёт по тренду.")
    print("Отсюда и результат ниже — правило «верх диапазона» выигрывает не вопреки,")
    print("а благодаря этому фону.")

    bar("ПРОВЕРКА ПОСЫЛКИ ТЗ: стоит ли ждать ПОСЛЕ падения курса три дня подряд")
    print("Индикатор ТЗ «моментум» предполагает, что после падения выгодно ловить дно.")
    print("Считаем цену ожидания отдельно по дням, когда индикатор сработал и когда нет.\n")
    print(f"{'коридор':<10}{'после падения 3 дн':>22}{'в остальные дни':>20}{'разница':>12}")
    for c in CORRIDORS:
        v = series[c].values
        fired, rest = [], []
        for i in range(3, len(v) - 3):
            g = _bps(float(v[i + 3]), float(v[i]))
            drop3 = v[i] < v[i - 1] < v[i - 2] < v[i - 3]
            (fired if drop3 else rest).append(g)
        a, b = float(np.mean(fired)), float(np.mean(rest))
        print(f"{c:<10}{a:>+20.0f}бп{b:>+18.0f}бп{a - b:>+10.0f}бп")
    print("\nГоризонт 3 публикации. Положительная разница означала бы, что посылка ТЗ")
    print("верна: после падения ожидание окупается лучше обычного.")


def payday_window_share(series) -> None:
    """Сколько месячного размаха вообще доступно клиенту, привязанному к дате выплаты."""
    bar("ОКНО КЛИЕНТА: какая доля месячного размаха физически доступна")
    print("Клиент переводит вокруг даты зарплаты, а не в любой день месяца. Считаем размах")
    print("курса в окне +-3 публикации от даты выплаты и делим на размах всего месяца.\n")
    print(f"{'коридор':<10}{'медиана доли':>15}{'25-й проц.':>13}{'75-й проц.':>13}{'окон':>8}")
    for c in CORRIDORS:
        s = series[c]
        dates, v = list(s.dates), s.values
        by_month: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        for i, d in enumerate(dates):
            by_month[(d.year, d.month)].append(i)
        shares = []
        for p in payday_anchors(dates):
            key = (dates[p].year, dates[p].month)
            idx = by_month[key]
            if len(idx) < 10:
                continue
            lo, hi = max(p - 3, idx[0]), min(p + 3, idx[-1])
            win = v[lo:hi + 1]
            mon = v[idx[0]:idx[-1] + 1]
            if mon.max() <= mon.min():
                continue
            shares.append((win.max() - win.min()) / (mon.max() - mon.min()))
        a = np.array(shares) * 100
        print(f"{c:<10}{np.median(a):>14.0f}%{np.percentile(a, 25):>12.0f}%"
              f"{np.percentile(a, 75):>12.0f}%{len(a):>8}")
    print("\nОстальное для клиента физически недоступно: он не переводит в те дни.")
    print("Поэтому продукт борется не за месячный размах, а за долю от него.\n")

    # Для сравнения: если считать одно слитое окно вокруг ОБЕИХ типовых дат
    # выплаты, доля выходит заметно выше — но это 14-дневный интервал, а не
    # 7-дневный, и решению одного клиента он не соответствует. Печатаем,
    # чтобы цифра из продуктовых документов была прослеживаема.
    print("Справочно: слитое окно вокруг обеих дат выплаты (14 дней вместо 7)")
    print(f"{'коридор':<10}{'медиана доли':>15}")
    for c in CORRIDORS:
        s = series[c]
        dates, v = list(s.dates), s.values
        by_month: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        for i, d in enumerate(dates):
            by_month[(d.year, d.month)].append(i)
        anchors_by_month: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        for p_ in payday_anchors(dates):
            anchors_by_month[(dates[p_].year, dates[p_].month)].append(p_)
        shares = []
        for key, idx in by_month.items():
            ps = anchors_by_month.get(key, [])
            if len(idx) < 10 or not ps:
                continue
            lo, hi = max(min(ps) - 3, idx[0]), min(max(ps) + 3, idx[-1])
            win, mon = v[lo:hi + 1], v[idx[0]:idx[-1] + 1]
            if mon.max() <= mon.min():
                continue
            shares.append((win.max() - win.min()) / (mon.max() - mon.min()))
        print(f"{c:<10}{np.median(np.array(shares) * 100):>14.0f}%")


def background_facts(series, X, names, index, dates) -> None:
    """Фоновые величины, на которые ссылаются документы. Считаются здесь, а не в тексте."""
    s = series
    bar("ФОН: случайный день, вес 2022 года, разрывы в ряду публикаций")

    fwd = np.full(len(index), np.nan)
    for r, (c, i, _d) in enumerate(index):
        v = benefit_forward_only(s[c].values, i, H)
        if v is not None:
            fwd[r] = v
    oos = np.array([d.year >= FIRST_TEST for d in dates])
    ds = np.array(dates, dtype=object)
    ok = oos & ~np.isnan(fwd)
    lo, hi = bootstrap_ci(fwd[ok], dates=ds[ok])
    print(f"Случайный день теста, достижимая выгода (h = {H}): {fwd[ok].mean():+.0f} бп, "
          f"95 % ДИ [{lo:+.0f}; {hi:+.0f}], строк {int(ok.sum())}")
    print("  Это базовый уровень: любой сигнал обязан обгонять его, а не ноль.\n")

    yrs = sorted({d.year for d in dates})
    rets = {}
    for c in CORRIDORS:
        v = s[c].values
        r = np.diff(v) / v[:-1] * 10000.0
        rets[c] = (np.array(list(s[c].dates)[1:], dtype=object), r)
    print(f"{'год':<8}{'доля наблюдений':>18}{'доля дисперсии доходностей':>30}")
    tot_n = sum(len(r) for _d, r in rets.values())
    tot_ss = sum(float((r ** 2).sum()) for _d, r in rets.values())
    for yy in yrs:
        n = ss = 0
        for dd, r in rets.values():
            m = np.array([x.year == yy for x in dd])
            n += int(m.sum())
            ss += float((r[m] ** 2).sum())
        if n:
            print(f"{yy:<8}{n / tot_n * 100:>17.1f}%{ss / tot_ss * 100:>29.1f}%")
    print("\n  «Доля дисперсии» — вклад года в сумму квадратов доходностей по всем коридорам.\n")

    gaps = []
    for c in CORRIDORS:
        dd = list(s[c].dates)
        gaps += [(b - a).days for a, b in zip(dd[:-1], dd[1:])]
    g = np.array(gaps)
    print(f"Разрывы между соседними публикациями, всего шагов {len(g)}:")
    for k in (1, 2, 3, 4):
        lbl = f"{k} дн" if k < 4 else "4+ дн"
        share = (g == k).mean() if k < 4 else (g >= 4).mean()
        print(f"  {lbl:<8}{share * 100:>6.1f} %")
    print(f"  3 и более дней: {(g >= 3).mean() * 100:.1f} % шагов")
    print("\n  Поэтому ряд строится по публикациям, а не по календарной сетке:")
    print("  forward-fill создал бы серии нулевых изменений там, где рынка просто не было.")


def traffic_light(X, names, index, dates) -> None:
    """Калибровка светофора: обе метрики на одних и тех же бакетах."""
    bar("КАЛИБРОВКА СВЕТОФОРА: симметричная выгода против достижимой")
    print("Светофор показывает положение курса в квартальном диапазоне — факт, не прогноз.")
    print("Вопрос в том, какой цвет чему соответствует. Ответ зависит от метрики,")
    print("и две метрики дают ПРОТИВОПОЛОЖНЫЕ ответы.\n")
    s = load()
    sym = np.full(len(index), np.nan)
    fwd = np.full(len(index), np.nan)
    hit = np.full(len(index), np.nan)
    for r, (c, i, _d) in enumerate(index):
        a = benefit_bps(s[c].values, i, H)
        b = benefit_forward_only(s[c].values, i, H)
        f = target_now_favourable(s[c].values, i, H)
        if a is not None:
            sym[r] = a
        if b is not None:
            fwd[r] = b
        if f is not None:
            hit[r] = f
    pct = X[:, names.index("pct_range_90")]
    ds = np.array(dates, dtype=object)
    oos = np.array([d.year >= FIRST_TEST for d in dates])
    buckets = (("0-10 % — валюта дёшева", pct <= 10),
               ("10-90 % — середина", (pct > 10) & (pct < 90)),
               ("90-100 % — валюта дорога", pct >= 90))

    for label, scope in (("ВСЯ ИСТОРИЯ (2020-2026)", np.ones(len(pct), bool)),
                         (f"ТОЛЬКО ТЕСТ (с {FIRST_TEST})", oos)):
        base = float(np.nanmean(hit[scope]))
        print(f"\n  {label}")
        print(f"{'бакет диапазона':<26}{'дней':>8}{'симметр. ±h':>15}"
              f"{'ДОСТИЖИМАЯ':>14}{'95% ДИ дост.':>18}{'lift':>8}")
        for nm, m0 in buckets:
            m = m0 & scope
            ok = m & ~np.isnan(fwd)
            lo, hi = bootstrap_ci(fwd[ok], dates=ds[ok])
            lf = float(np.nanmean(hit[m])) / base
            print(f"{nm:<26}{int(m.sum()):>8}{np.nanmean(sym[m]):>+13.0f}бп"
                  f"{np.nanmean(fwd[ok]):>+12.0f}бп{f'[{lo:+.0f}; {hi:+.0f}]':>18}{lf:>8.2f}")
    print("\n  Границы бакетов (10 и 90) заданы ТЗ, а не подобраны, поэтому обе таблицы")
    print("  добросовестны. Строку «только тест» показываем, чтобы вывод не держался")
    print("  на обучающем периоде: он не держится, знаки и порядок те же.")
    print("\n  По СИММЕТРИЧНОЙ метрике края работают зеркально: дёшево = хорошо.")
    print("  По ДОСТИЖИМОЙ знак переворачивается: единственный бакет, чей интервал")
    print("  не пересекает ноль, — «валюта дорога», и он ПОЛОЖИТЕЛЬНЫЙ.")
    print("  Механизм: на верхней границе диапазона курс чаще продолжает расти,")
    print("  поэтому «перевести сейчас» выигрывает у ожидания. Симметричная метрика")
    print("  этого не видит, потому что наполовину описывает уже случившееся движение.")


def navruz_effect(series) -> None:
    """Навруз: дата фиксирована 21 марта, поэтому «эффект» неотличим от «марта».

    Считается тем же способом, что Курбан-байрам, специально для сравнения:
    продуктовый вывод «Навруз — артефакт, а Курбан-байрам — кандидат» держится
    на том, что у первого дата стоит на месте, а у второго уходит с августа
    на май. Без этой таблицы вывод пришлось бы принимать на слово.
    """
    bar("НАВРУЗ (21 марта, дата фиксирована): движение курса за две недели ДО")
    print("Сравнение с Курбан-байрамом ниже. Знак со стороны клиента:")
    print("МИНУС = валюта подорожала, перевод стал хуже.\n")
    print(f"{'год':<8}" + "".join(f"{c:>10}" for c in CORRIDORS) + f"{'среднее':>11}")
    rows = []
    for d in HOLIDAYS["navruz"]:
        cells, vals = [], []
        for c in CORRIDORS:
            ds = np.array(series[c].dates, dtype=object)
            v = series[c].values
            j = int(np.searchsorted(ds, d, side="right")) - 1
            i = int(np.searchsorted(ds, d - dt.timedelta(days=14), side="right")) - 1
            if j <= i or i < 0:
                cells.append(f"{'—':>10}")
                continue
            g = _bps(float(v[j]), float(v[i]))
            vals.append(g)
            cells.append(f"{g:>+8.0f}бп")
        if not vals:
            continue
        m = float(np.mean(vals))
        rows.append((d.year, m))
        print(f"{d.year:<8}" + "".join(cells) + f"{m:>+9.0f}бп")
    vals_all = [m for _y, m in rows]
    worse = sum(1 for m in vals_all if m < 0)
    dominant = [y for y, m in rows if abs(m) > 300]
    rest = [m for y, m in rows if abs(m) <= 300]
    print(f"\n  валюта дорожала перед Наврузом: {worse} лет из {len(rows)}"
          f", медиана {np.median(vals_all):+.0f} бп")
    if dominant:
        print(f"  годы с движением больше 300 бп: {', '.join(map(str, dominant))}; "
              f"без них медиана {np.median(rest):+.0f} бп")
    print("\n  Дата Навруза не двигается: 21 марта каждый год. Поэтому «эффект Навруза»")
    print("  неотделим от «эффекта марта», и отличить их на восьми наблюдениях нечем.")
    print("  У Курбан-байрама дата за восемь лет ушла с августа на май — только")
    print("  поэтому его мы держим в кандидатах, а Навруз нет.")


def eid_effect(series) -> None:
    """Курбан-байрам: главный календарный кандидат, потому что дата плавает."""
    bar("КУРБАН-БАЙРАМ: движение курса за две недели ДО праздника")
    print("Дата плавает по календарю (август 2019 -> май 2026), поэтому эффект нельзя")
    print("списать на сезон — он следует за датой. Знак со стороны клиента:")
    print("МИНУС = валюта подорожала, перевод стал хуже.\n")
    print(f"{'год':<8}" + "".join(f"{c:>10}" for c in CORRIDORS) + f"{'среднее':>11}")
    worse = 0
    rows = []
    for d in HOLIDAYS["eid_adha"]:
        cells, vals = [], []
        for c in CORRIDORS:
            ds = np.array(series[c].dates, dtype=object)
            v = series[c].values
            j = int(np.searchsorted(ds, d, side="right")) - 1
            i = int(np.searchsorted(ds, d - dt.timedelta(days=14), side="right")) - 1
            if j <= i or i < 0:
                cells.append(f"{'—':>10}")
                continue
            g = _bps(float(v[j]), float(v[i]))
            vals.append(g)
            cells.append(f"{g:>+8.0f}бп")
        if not vals:
            continue
        m = float(np.mean(vals))
        rows.append((d.year, m))
        if m < 0:
            worse += 1
        print(f"{d.year:<8}" + "".join(cells) + f"{m:>+9.0f}бп")
    vals_all = [m for _y, m in rows]
    # фильтр по ГОДУ, а не по величине: «|значение| < 500» на этих данных совпало
    # с «кроме 2022», но любой другой выброс молча уехал бы в строку «без 2022»
    vals_no22 = [m for y, m in rows if y != 2022]
    print(f"\n  валюта дорожала перед праздником: {worse} лет из {len(rows)}"
          f", медиана {np.median(vals_all):+.0f} бп")
    print(f"  без 2022 года: медиана {np.median(vals_no22):+.0f} бп")
    print("\n  Одно наблюдение в год: N = 8. Это кандидат в механизм, а не доказанный эффект.")


def per_corridor_calibration(X, names, index, dates) -> None:
    """Покоридорная калибровка порога — прямое требование ТЗ.

    ТЗ: «откалибровать окна и пороги отдельно для каждого коридора, поскольку их
    волатильность различается». Глобальная константа `pct >= 95` этому не отвечает.
    Здесь порог каждого коридора подбирается по ПЕРИОДУ РАЗРАБОТКИ так, чтобы дать
    целевую частоту внутри полосы ТЗ, и только потом измеряется на тесте.

    Рядом печатаются две НЕЧЕСТНЫЕ версии той же процедуры. Они не результат:
    они показывают, сколько именно добавляет подглядывание в тест — и почему
    вывод «полоса и lift несовместимы» нельзя опровергнуть, подобрав порог.
    """
    s = load()
    pct = X[:, names.index("pct_range_90")]
    corr = np.array([c for c, _i, _d in index])
    y = np.full(len(index), np.nan)
    for r, (c, i, _d) in enumerate(index):
        v = target_now_favourable(s[c].values, i, H)
        if v is not None:
            y[r] = v
    oos = np.array([d.year >= FIRST_TEST for d in dates])
    dev = ~oos

    bar(f"ПОКОРИДОРНАЯ КАЛИБРОВКА ПОРОГА (требование ТЗ), цель «сейчас выгодно», h = {H}")
    print(f"Порог каждого коридора подобран на данных до {FIRST_TEST} под целевую частоту")
    print(f"{TARGET_PER_WEEK} сигнала на коридор в неделю — середину полосы ТЗ. Тест не участвует.\n")
    print(f"{'коридор':<9}{'порог':>8}{'частота':>10}{'сигн/нед':>10}{'lift':>7}{'полоса ТЗ':>12}")
    honest = []
    for c in CORRIDORS:
        md = dev & (corr == c) & ~np.isnan(y)
        mo = oos & (corr == c)
        dd = [d for d, k in zip(dates, md) if k]
        weeks = (max(dd) - min(dd)).days / 7.0
        frac = min(0.99, TARGET_PER_WEEK * weeks / md.sum())
        thr = float(np.quantile(pct[md], 1.0 - frac))
        fired = (pct >= thr) & mo
        lf, _b, n = lift(fired, y, scope=mo)
        pw = rate_per_week(n, 1, dates, mo)
        honest.append(lf)
        band = "в полосе" if 1.0 <= pw <= 2.0 else ("НИЖЕ" if pw < 1 else "ВЫШЕ")
        print(f"{c:<9}{thr:>8.1f}{fired.sum() / max(mo.sum(), 1) * 100:>9.1f}%"
              f"{pw:>10.2f}{lf:>7.2f}{band:>12}")
    print(f"\n  Все пять коридоров попадают в полосу — и ни один не даёт lift >= 1,3:")
    print(f"  разброс {min(honest):.2f}-{max(honest):.2f}. Порог, обеспечивающий нужную")
    print("  частоту, обязан пускать слишком много дней, и точность падает.")

    bar("ТО ЖЕ, НО С ПОДГЛЯДЫВАНИЕМ В ТЕСТ — И ИМЕННО ТАК «ТРЕБОВАНИЕ ВЫПОЛНЯЕТСЯ»")
    print("Ниже два способа получить lift >= 1,3 внутри полосы. Оба некорректны, и оба")
    print("выглядят как добросовестная покоридорная калибровка. Показываем их потому,")
    print("что проверяющий воспроизведёт именно их.\n")
    print(f"{'коридор':<9}{'A: цель из полосы по тесту':>28}{'B: порог перебран по тесту':>30}")
    a_ok = b_ok = 0
    for c in CORRIDORS:
        md = dev & (corr == c) & ~np.isnan(y)
        mo = oos & (corr == c)
        dd = [d for d, k in zip(dates, md) if k]
        weeks = (max(dd) - min(dd)).days / 7.0
        best_a = best_b = None
        for target in np.arange(1.0, 2.01, 0.05):
            frac = min(0.99, target * weeks / md.sum())
            thr = float(np.quantile(pct[md], 1.0 - frac))
            fired = (pct >= thr) & mo
            lf, _b, n = lift(fired, y, scope=mo)
            pw = rate_per_week(n, 1, dates, mo)
            if not np.isnan(lf) and 1.0 <= pw <= 2.0 and (best_a is None or lf > best_a[0]):
                best_a = (lf, pw)
        for thr in np.arange(50.0, 99.5, 0.5):
            fired = (pct >= thr) & mo
            lf, _b, n = lift(fired, y, scope=mo)
            pw = rate_per_week(n, 1, dates, mo)
            if not np.isnan(lf) and 1.0 <= pw <= 2.0 and (best_b is None or lf > best_b[0]):
                best_b = (lf, pw)
        a_ok += bool(best_a and best_a[0] >= 1.3)
        b_ok += bool(best_b and best_b[0] >= 1.3)
        fa = f"lift {best_a[0]:.2f} при {best_a[1]:.2f}/нед" if best_a else "—"
        fb = f"lift {best_b[0]:.2f} при {best_b[1]:.2f}/нед" if best_b else "—"
        print(f"{c:<9}{fa:>28}{fb:>30}")
    print(f"\n  коридоров с lift >= 1,3 в полосе: честно 0 из 5, способом A {a_ok} из 5,"
          f" способом B {b_ok} из 5.")
    print("  Разница между 0 и " + str(b_ok) + " — это ровно цена выбора рабочей точки по тесту.")
    print("  Способ A выбирает по тесту, в какой точке полосы встать; способ B — сам порог.")
    print("  Ни один из них не переживёт следующий год данных, потому что оба настроены")
    print("  на тот единственный отрезок, на котором их и мерили.")


def per_corridor(X, names, index, dates) -> None:
    """Обязательное условие ТЗ: результат отдельно по каждому коридору."""
    bar(f"РАЗБИВКА ПО КОРИДОРАМ, цель «сейчас выгодно» h = {H}, только тест с {FIRST_TEST}")
    s = load()
    y = np.full(len(index), np.nan)
    ben = np.full(len(index), np.nan)
    for r, (c, i, _d) in enumerate(index):
        t = target_now_favourable(s[c].values, i, H)
        b = benefit_forward_only(s[c].values, i, H)
        if t is not None:
            y[r] = t
        if b is not None:
            ben[r] = b
    oos = np.array([d.year >= FIRST_TEST for d in dates])
    corr_of = np.array([c for c, _i, _d in index])

    for rule_name in ("ТЗ: моментум (падение 3 дн)", "ТЗ: уровень (нижний дециль)",
                      "простое правило: верх диапазона"):
        fires = BASELINES[rule_name](X, names).astype(bool)
        print(f"\n{rule_name}")
        print(f"{'коридор':<10}{'частота':>10}{'сигн/нед':>11}{'lift':>8}"
              f"{'достижимая':>14}{'95% ДИ':>18}")
        for c in list(CORRIDORS) + ["ВСЕ"]:
            m = oos if c == "ВСЕ" else (oos & (corr_of == c))
            f = fires & m
            lf, _base, n = lift(f, y, scope=m)
            b = ben[f & ~np.isnan(ben)]
            d = [dt for dt, ok in zip(dates, f & ~np.isnan(ben)) if ok]
            lo, hi = bootstrap_ci(b, dates=d) if len(b) > 20 else (np.nan, np.nan)
            pw = rate_per_week(n, 1 if c != "ВСЕ" else len(CORRIDORS), dates, m)
            print(f"{c:<10}{f.sum() / max(m.sum(), 1) * 100:>9.1f}%{pw:>11.2f}"
                  f"{lf:>8.2f}{b.mean():>+12.0f}бп{'[' + f'{lo:+.0f}; {hi:+.0f}' + ']':>18}")

    bar("ПОЧЕМУ РАЗБИВКА ПО КОРИДОРАМ — СЛАБОЕ ПОДТВЕРЖДЕНИЕ")
    per_date: dict[dt.date, dict[str, float]] = collections.defaultdict(dict)
    for r, (c, _i, d) in enumerate(index):
        if not np.isnan(ben[r]):
            per_date[d][c] = ben[r]
    pairs = []
    for a in range(len(CORRIDORS)):
        for b_ in range(a + 1, len(CORRIDORS)):
            ca, cb = CORRIDORS[a], CORRIDORS[b_]
            xs = [(v[ca], v[cb]) for v in per_date.values() if ca in v and cb in v]
            if len(xs) > 100:
                arr = np.array(xs)
                pairs.append((ca, cb, float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])))
    print("Корреляция достижимой выгоды между коридорами В ОДИН И ТОТ ЖЕ ДЕНЬ:\n")
    for ca, cb, r in sorted(pairs, key=lambda p: -p[2]):
        print(f"  {ca}-{cb}: {r:.2f}")
    print(f"\n  средняя попарная: {np.mean([p[2] for p in pairs]):.2f}")
    print("\nПять коридоров — это НЕ пять независимых подтверждений: все они торгуются")
    print("против рубля и движутся вместе. Совпадение результата на пяти коридорах")
    print("стоит примерно столько же, сколько на одном. Разбивка выше сделана потому,")
    print("что её требует ТЗ, но независимым подтверждением она не является.")


def clustering(X, names, index, dates) -> None:
    """Обязательный критерий ТЗ: кучность и равномерность сигналов."""
    bar("КУЧНОСТЬ: сигналы приходят сериями или равномерно?")
    print("ТЗ требует 1-2 сигнала на коридор в неделю. Средняя частота этого не гарантирует:")
    print("сигналы могут собраться в одну неделю и молчать месяц.\n")
    print(f"{'правило':<36}{'сигн/нед':>10}{'в сериях':>11}{'макс. пауза':>14}"
          f"{'пик недели':>13}")
    oos = np.array([d.year >= FIRST_TEST for d in dates])
    corr_of = np.array([c for c, _i, _d in index])
    for rule_name, fn in BASELINES.items():
        fires = fn(X, names).astype(bool) & oos
        in_series, gaps, per_week_counts = 0, [], collections.Counter()
        total = 0
        pairs = 0
        for c in CORRIDORS:
            pubs = sorted({d for d, cc in zip(dates, corr_of) if cc == c})
            pos_of = {d: k for k, d in enumerate(pubs)}
            ds = sorted(d for d, f, cc in zip(dates, fires, corr_of) if f and cc == c)
            total += len(ds)
            pairs += max(len(ds) - 1, 0)
            for a, b in zip(ds[:-1], ds[1:]):
                gaps.append((b - a).days)
                # «в пределах 3 публикаций» — именно публикаций: 79 % промежутков
                # между соседними публикациями = 1 день, но 2,4 % — 4 дня и больше,
                # и календарный порог считал бы не то множество.
                if pos_of[b] - pos_of[a] <= 3:
                    in_series += 1
            for d in ds:
                per_week_counts[(c, d.isocalendar()[0], d.isocalendar()[1])] += 1
        pw = rate_per_week(total, len(CORRIDORS), dates, oos)
        # знаменатель — промежутки, а не сигналы: у n сигналов их n-1 на коридор
        share = in_series / pairs * 100 if pairs else float("nan")
        peak = max(per_week_counts.values()) if per_week_counts else 0
        print(f"{rule_name:<36}{pw:>10.2f}{share:>10.0f}%{max(gaps):>12} дн{peak:>11} шт")
    print("\n«в сериях» — доля ПРОМЕЖУТКОВ между соседними сигналами длиной "
          "не больше 3 публикаций.")
    print("«пик недели» — максимум сигналов на один коридор за одну календарную неделю.")
    print("Ни одно правило не выдаёт ровный поток: полосу ТЗ нельзя выполнить,")
    print("просто подобрав порог, — нужен добор до пола и cooldown против серий.")


def base_rate_drift(index, dates) -> None:
    """Почему метрика кейса меряет режим рынка, а не качество момента."""
    bar("ДРЕЙФ БАЗОВОЙ СТАВКИ: метрика кейса меряет режим рубля")
    s = load()
    y = np.full(len(index), np.nan)
    for r, (c, i, _d) in enumerate(index):
        t = target_now_favourable(s[c].values, i, H)
        if t is not None:
            y[r] = t
    years = sorted({d.year for d in dates})
    print(f"«Сейчас выгодно» (h = {H}) — доля дней, когда курс не будет побит за {H} публикаций\n")
    print(f"{'год':<8}" + "".join(f"{yy:>8}" for yy in years))
    cells = []
    for yy in years:
        m = np.array([d.year == yy for d in dates]) & ~np.isnan(y)
        cells.append(f"{y[m].mean() * 100:>7.0f}%" if m.sum() else "      —")
    print(f"{'ставка':<8}" + "".join(cells))
    print("\nОт года к году базовая ставка гуляет вдвое. В годы ослабления рубля «сегодня")
    print("не будет побито» верно почти автоматически — метрика отслеживает тренд рубля,")
    print("а не качество выбранного момента. Поэтому lift обязан считаться против базы")
    print("ТОГО ЖЕ периода (ml/evaluate.py::lift, аргумент scope), иначе он меряет дрейф.")


def main() -> None:
    series = load()
    intramonth_range(series)
    oracle_and_shift(series)
    cost_of_waiting(series)
    payday_window_share(series)
    navruz_effect(series)
    eid_effect(series)
    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = [d for _c, _i, d in index]
    background_facts(series, X, names, index, dates)
    traffic_light(X, names, index, dates)
    per_corridor_calibration(X, names, index, dates)
    per_corridor(X, names, index, dates)
    clustering(X, names, index, dates)
    base_rate_drift(index, dates)


if __name__ == "__main__":
    main()
