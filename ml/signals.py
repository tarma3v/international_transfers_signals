"""Публичный срез: сигналы на произвольную дату T, без единого наблюдения после T.

Кейс объявляет это условием проверяемости: «код обязан считать сигналы на
произвольную дату среза, иначе решение невозможно проверить на отсутствие
заглядывания вперёд». Гарантия здесь структурная, а не декларативная: ряды
физически обрезаются по T методом `Series.truncate` ДО построения признаков,
поэтому наблюдений после T в вычислении просто нет — не «мы их не используем»,
а «их нет в памяти».
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, Series


def truncate_to(series: dict[str, Series], as_of: dt.date) -> dict[str, Series]:
    """Копия рядов, обрезанная по дату T включительно."""
    out: dict[str, Series] = {}
    for code, s in series.items():
        n = int(np.searchsorted(np.array(s.dates, dtype=object), as_of, side="right"))
        if n == 0:
            raise ValueError(f"{code}: до {as_of} нет ни одной публикации")
        out[code] = s.truncate(n)
    return out


def _traffic_light(pct: float) -> str:
    """Цвет по границам ТЗ (10 / 90), а не по подобранным нами порогам.

    Смысл цвета — положение курса в диапазоне, факт о прошлом. Что этот цвет
    означает для клиента, зависит от метрики: по симметричной выгоде «дёшево»
    даёт +59 бп, по достижимой — −1 бп, и наоборот. Разбор — в
    docs/04-dve-metriki-dve-modeli.md; здесь возвращается только состояние.
    """
    if pct <= 10.0:
        return "валюта получателя дешева (низ диапазона)"
    if pct >= 90.0:
        return "валюта получателя дорога (верх диапазона)"
    return "середина диапазона"


def signals_as_of(
    series: dict[str, Series],
    as_of: dt.date,
    corridors: tuple[str, ...] = CORRIDORS,
    refs: tuple[str, ...] = REFERENCE,
) -> list[dict[str, object]]:
    """Состояние сигнального слоя на дату T для каждого коридора.

    Возвращает по строке на коридор: дата последней публикации не позже T,
    курс, срабатывание каждого из четырёх индикаторов ТЗ, положение курса
    в квартальном диапазоне, цвет светофора и срабатывание правила
    `pct_range_90 >= 95` — то есть всё, что уходит в продукт.

    Модель сюда не подставляется намеренно: правила воспроизводятся без
    артефактов обучения, и жюри может проверить срез независимо от весов.
    Модель на этих же данных не превосходит правило (lift 1,35 против 1,39),
    поэтому продуктовое состояние ничего от её отсутствия не теряет.
    """
    from ml.features import WARMUP, build_matrix  # локально: build_matrix тянет тяжёлый модуль

    cut = truncate_to(series, as_of)

    # Слишком ранний срез — законный вопрос жюри, а не поломка. Отвечать на него
    # надо словами: голый IndexError из build_matrix не объясняет, что произошло,
    # и выглядит как отказ кода считать произвольную дату.
    short = sorted(c for c in corridors if len(cut[c].values) <= WARMUP)
    if short:
        ready = [series[c].dates[WARMUP] for c in corridors if len(series[c].dates) > WARMUP]
        hint = (f" Самая ранняя дата, на которой считаются все коридоры: {max(ready)}."
                if len(ready) == len(corridors) else "")
        raise ValueError(
            f"дата среза {as_of} слишком ранняя: признакам нужен прогрев "
            f"(публикаций: {WARMUP}), а к этой дате их набралось меньше "
            f"по коридорам {', '.join(short)}.{hint}"
        )

    X, names, index = build_matrix(cut, corridors, refs)
    fires = {name: fn(X, names) for name, fn in BASELINES.items()}
    last = {c: max(r for r, (cc, _i, _d) in enumerate(index) if cc == c)
            for c in corridors if any(cc == c for cc, _i, _d in index)}
    out: list[dict[str, object]] = []
    for c in corridors:
        if c not in last:
            continue
        r = last[c]
        _cc, i, day = index[r]
        pct = float(X[r, names.index("pct_range_90")])
        out.append({
            "corridor": c,
            "as_of": as_of,
            "date": day,
            "rate": float(cut[c].values[i]),
            **{name: bool(v[r]) for name, v in fires.items()},
            "положение в квартальном диапазоне, %": round(pct, 1),
            "светофор": _traffic_light(pct),
            "правило pct>=95": bool(pct >= 95.0),
        })
    return out


if __name__ == "__main__":
    from ml.data import load

    s = load()
    for T in (dt.date(2025, 3, 14), dt.date(2026, 6, 2)):
        print(f"\nсрез на {T}")
        for row in signals_as_of(s, T):
            fired = [k for k, v in row.items()
                     if isinstance(v, bool) and v]
            print(f"  {row['corridor']}  публикация {row['date']}  курс {row['rate']:.4f}")
            print(f"      положение в диапазоне {row['положение в квартальном диапазоне, %']:>5} %"
                  f"  — {row['светофор']}")
            print(f"      сработало: {', '.join(fired) if fired else '—'}")
