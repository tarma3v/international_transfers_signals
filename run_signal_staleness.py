"""Устаревание сигнала: сколько живёт срабатывание и когда пуш пора гасить.

Развилка кейса, которой у нас не было. Между расчётом сигнала и открытием пуша
проходит время: ЦБ публикует курс после 15:30 МСК, пуш уходит вечером, а клиент
открывает его вечером, утром следующего дня или через неделю. Вопрос продукта:
до какого момента сообщение остаётся правдой и что делать, когда перестало.

ПРАВИЛО РЕШЕНИЯ ЗАДАНО ДО ПРОГОНА и здесь ровно одно. Рекомендация живёт до
задержки k, если одновременно:
  1) средний остаток достижимой выгоды не ниже порога запуска LAUNCH_BPS;
  2) интервал РАЗНИЦЫ «сигнал минус случайный день» не накрывает ноль на
     уровне, поправленном на число проверяемых задержек (Бонферрони).
Второе условие — про разницу, а не про сам остаток: «сигнал равен фону» из
непопадания точки фона в интервал сигнала не следует, это подмена отсутствия
доказательства доказательством отсутствия.

ЧЕТЫРЕ ТРЕБОВАНИЯ К ЧЕСТНОМУ СЧЁТУ, каждое стоило отдельной ошибки:

1. ОДНА КОГОРТА НА ВСЕ СТРОКИ. Отсечка «горизонт не помещается» своя у каждой
   задержки, поэтому строки таблицы легко оказываются про разные наборы
   сигналов, а сравнивать их между собой уже нельзя. Берём сигналы, у которых
   помещается САМАЯ ДЛИННАЯ задержка, и все строки считаем по ним.
2. БЛОЧНЫЙ БУТСТРАП. benefit_forward_only(i) смотрит на v[i+1 .. i+H], поэтому
   соседние публикации делят H−1 будущих курсов, а срабатывания идут сериями
   (в данных есть серия из 32 публикаций подряд). Ресемплирование отдельных
   дней рвёт эту зависимость и даёт интервалы уже настоящих. Ресемплируем
   блоки подряд идущих дат длиной H.
3. РАЗНИЦА, А НЕ ДВА ИНТЕРВАЛА. Сигнал и фон считаются на одних и тех же датах
   в одной и той же бутстрап-реплике, поэтому интервал строится сразу для их
   разницы и учитывает общий рыночный сдвиг.
4. РАЗБИВКА ПО ГОДАМ И КОРИДОРАМ. Общий срок жизни, применённый ко всем,
   оставляет пуш живым там, где по своему коридору он уже не проходит порог.
   Печатаем разбивку рядом с общим числом, а не вместо него.

Ограничение, которое надо назвать вслух: ряд ЦБ дневной, поэтому мельче одной
публикации мы не видим. Внутридневное устаревание — вопрос к биржевому ряду
MOEX, здесь он не решается.

Запуск: PYTHONPATH=. python run_signal_staleness.py
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.features import build_matrix
from ml.targets import benefit_forward_only

RULE = "простое правило: верх диапазона"
H = 5                       # горизонт выгоды, публикаций
DELAYS = (0, 1, 2, 3, 5)    # задержка между сигналом и действием, публикаций
FIRST_TEST = 2021           # тест начинается здесь, обучение сюда не заглядывает
LAUNCH_BPS = 30.0           # порог запуска из продуктовых чисел: +30 бп
B = 4000                    # реплик бутстрапа
SEED = 0
# Бонферрони на число проверяемых задержек: пять маргинальных 95 % интервалов
# дают одновременное покрытие сильно ниже заявленного.
ALPHA = 0.05 / len(DELAYS)


def _bps(new: float, old: float) -> float:
    """Со стороны клиента: плюс = курс упал = за те же рубли дают больше."""
    return -(new - old) / old * 10000.0


def bar(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def block_bootstrap_diff(
    sig_sum: np.ndarray, sig_n: np.ndarray,
    bg_sum: np.ndarray, bg_n: np.ndarray,
    block: int, alpha: float, seed: int = SEED,
) -> tuple[float, float]:
    """Интервал разницы средних «сигнал минус фон» блочным бутстрапом по датам.

    Все четыре массива выровнены по одному списку дат. Ресемплируются БЛОКИ
    подряд идущих дат: перекрывающиеся форвардные окна делают соседние дни
    зависимыми, и построчный бутстрап это игнорирует.
    """
    n = len(bg_n)
    if n < 10 or sig_n.sum() < 10:
        return float("nan"), float("nan")
    starts_n = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    idx = np.arange(n)
    for b in range(B):
        starts = rng.integers(0, n, starts_n)
        pick = np.concatenate([idx[s:s + block] for s in starts])
        s_n = sig_n[pick].sum()
        if s_n == 0:
            out[b] = np.nan
            continue
        out[b] = sig_sum[pick].sum() / s_n - bg_sum[pick].sum() / bg_n[pick].sum()
    out = np.sort(out[~np.isnan(out)])
    if len(out) < 100:
        return float("nan"), float("nan")
    lo = float(out[int(alpha / 2 * (len(out) - 1))])
    hi = float(out[int((1 - alpha / 2) * (len(out) - 1))])
    return lo, hi


def main() -> None:
    series = load()
    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    fires = BASELINES[RULE](X, names).astype(bool)
    pct = X[:, names.index("pct_range_90")]
    row_of = {(c, i): row for row, (c, i, _d) in enumerate(index)}
    kmax = max(DELAYS)

    # ── фон: все тестовые строки, сгруппированные по дате действия ──────────
    bg_by_date: dict[object, list[float]] = defaultdict(list)
    for _row, (c, i, d) in enumerate(index):
        if d.year < FIRST_TEST:
            continue
        b = benefit_forward_only(series[c].values, i, H)
        if b is not None:
            bg_by_date[d].append(b)
    dates = sorted(bg_by_date)
    date_pos = {d: p for p, d in enumerate(dates)}
    bg_sum = np.array([sum(bg_by_date[d]) for d in dates])
    bg_n = np.array([len(bg_by_date[d]) for d in dates], dtype=float)
    bg_mean = bg_sum.sum() / bg_n.sum()

    # ── когорта: сигналы, у которых помещается самая длинная задержка ───────
    cohort = [
        (row, c, i, d) for row, (c, i, d) in enumerate(index)
        if fires[row] and d.year >= FIRST_TEST
        and i + kmax + H < len(series[c].values)
    ]

    bar(f"УСТАРЕВАНИЕ СИГНАЛА: правило «{RULE}», тест с {FIRST_TEST}")
    print("Задержка меряется в ПУБЛИКАЦИЯХ, а не в календарных днях: курс")
    print("публикуется по рабочим дням, и выходные не создают новых сигналов.")
    print(f"Когорта одна на все строки: {len(cohort)} срабатываний, у которых")
    print(f"помещается задержка +{kmax} и горизонт h={H}. Иначе строки таблицы")
    print("оказались бы про разные наборы сигналов.")
    print(f"Фон — случайный день теста: {bg_mean:+.0f} бп, {int(bg_n.sum())} строк.\n")

    print(f"{'задержка':<12}{'условие держится':>18}{'курс за задержку':>18}"
          f"{'остаток выгоды':>16}{'разница с фоном':>17}"
          f"{f'{100*(1-ALPHA):.0f}% ДИ разницы':>20}")

    rows_out = []
    for k in DELAYS:
        held, moves, benefits = 0, [], []
        s_sum = np.zeros(len(dates))
        s_n = np.zeros(len(dates))
        for _row, c, i, _d in cohort:
            v = series[c].values
            later = i + k
            later_row = row_of.get((c, later))
            if later_row is not None and pct[later_row] >= 95.0:
                held += 1
            moves.append(_bps(float(v[later]), float(v[i])))
            b = benefit_forward_only(v, later, H)
            if b is None:
                continue
            benefits.append(b)
            # кластер — дата ДЕЙСТВИЯ: с ней же сопоставляется фон
            p = date_pos.get(series[c].dates[later])
            if p is not None:
                s_sum[p] += b
                s_n[p] += 1
        mean_ben = float(np.mean(benefits))
        diff = mean_ben - bg_mean
        lo, hi = block_bootstrap_diff(s_sum, s_n, bg_sum, bg_n, H, ALPHA)
        med_move = float(np.median(moves))
        rows_out.append((k, 100.0 * held / len(cohort), med_move, mean_ben, diff, lo, hi))
        label = "в день сигнала" if k == 0 else f"+{k} публ."
        print(f"{label:<12}{100.0 * held / len(cohort):>17.0f}%{med_move:>+17.0f}бп"
              f"{mean_ben:>+14.0f}бп{diff:>+15.0f}бп"
              f"{f'[{lo:+.0f}; {hi:+.0f}]':>20}")

    print("\n  «условие держится» — правило всё ещё срабатывает на k-й публикации.")
    print("  «курс за задержку» — медиана движения курса со стороны клиента:")
    print("      плюс = подождать оказалось выгоднее, минус = клиент потерял.")
    print("  «остаток выгоды» — средняя достижимая выгода при действии на k-й")
    print(f"      публикации, горизонт h={H}. «разница с фоном» — она же минус")
    print("      случайный день; интервал построен блочным бутстрапом по датам")
    print(f"      (блок {H} дат) и поправлен на {len(DELAYS)} задержки по Бонферрони.")

    # ── правило решения, заданное до прогона ────────────────────────────────
    alive = [k for k, _h, _m, ben, _d, lo, hi in rows_out
             if ben >= LAUNCH_BPS and not (np.isnan(lo) or lo <= 0.0 <= hi)]
    bar("СРОК ЖИЗНИ РЕКОМЕНДАЦИИ")
    if alive:
        ttl = max(alive)
        row = next(r for r in rows_out if r[0] == ttl)
        print(f"  Оба условия выполняются до +{ttl} публикаций включительно:")
        print(f"  остаток {row[3]:+.0f} бп не ниже порога +{LAUNCH_BPS:.0f},"
              f" разница с фоном {row[4]:+.0f} бп,")
        print(f"  интервал [{row[5]:+.0f}; {row[6]:+.0f}] ноль не накрывает.")
    else:
        ttl = None
        print("  Ни одна задержка не проходит оба условия. Срок жизни не установлен.")
    first_fail = [r for r in rows_out if r[0] > (ttl if ttl is not None else -1)]
    if first_fail:
        k, _h, _m, ben, diff, lo, hi = first_fail[0]
        why = []
        if ben < LAUNCH_BPS:
            why.append(f"остаток {ben:+.0f} бп ниже порога +{LAUNCH_BPS:.0f}")
        if np.isnan(lo) or lo <= 0.0 <= hi:
            why.append(f"интервал разницы [{lo:+.0f}; {hi:+.0f}] накрывает ноль")
        print(f"\n  На +{k} условие уже нарушено: {', и '.join(why)}.")
        print("  Это значит «различить не можем», а не «сигнал доказанно равен")
        print("  фону»: доказательство равенства потребовало бы заранее заданной")
        print("  зоны эквивалентности и другого теста.")

    # ── разбивка: общий срок нельзя применять, не показав, где он не держится ─
    bar("ГДЕ ОБЩИЙ СРОК НЕ ДЕРЖИТСЯ: по годам и по коридорам")
    by_year: dict[int, dict[int, list[float]]] = {k: defaultdict(list) for k in DELAYS}
    by_corr: dict[int, dict[str, list[float]]] = {k: defaultdict(list) for k in DELAYS}
    for _row, c, i, d in cohort:
        for k in DELAYS:
            b = benefit_forward_only(series[c].values, i + k, H)
            if b is not None:
                by_year[k][d.year].append(b)
                by_corr[k][c].append(b)

    years = sorted(by_year[0])
    print(f"{'':<10}" + "".join(f"{y:>10}" for y in years))
    for k in DELAYS:
        cells = "".join(f"{np.mean(by_year[k][y]):>+9.0f}" + "бп" for y in years)
        print(f"{('+' + str(k) + ' публ.'):<10}{cells}")
    print(f"{'строк':<10}" + "".join(f"{len(by_year[0][y]):>10}" for y in years))

    corrs = sorted(by_corr[0])
    print(f"\n{'':<10}" + "".join(f"{c:>10}" for c in corrs))
    for k in DELAYS:
        cells = "".join(f"{np.mean(by_corr[k][c]):>+9.0f}" + "бп" for c in corrs)
        print(f"{('+' + str(k) + ' публ.'):<10}{cells}")
    print(f"{'строк':<10}" + "".join(f"{len(by_corr[0][c]):>10}" for c in corrs))

    if ttl is not None:
        weak_c = [c for c in corrs if np.mean(by_corr[ttl][c]) < LAUNCH_BPS]
        weak_y = [y for y in years if np.mean(by_year[ttl][y]) < 0]
        print(f"\n  На сроке +{ttl} общий остаток выше порога, но:")
        if weak_c:
            print("  — по коридорам ниже порога: "
                  + ", ".join(f"{c} ({np.mean(by_corr[ttl][c]):+.0f} бп)" for c in weak_c))
        if weak_y:
            print("  — годы с отрицательным остатком: "
                  + ", ".join(f"{y} ({np.mean(by_year[ttl][y]):+.0f} бп)" for y in weak_y))
        print("  Общий срок жизни к ним применять нельзя: там пуш не проходит")
        print("  собственный экономический порог, и срок должен быть покоридорным.")

    print("\n  Ограничение: ряд ЦБ дневной. Внутридневного устаревания — между")
    print("  отправкой вечером и открытием ночью — эти данные не показывают.")


if __name__ == "__main__":
    main()
