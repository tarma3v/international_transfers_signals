"""Признаки, вычисляемые ТОЛЬКО по прошлому.

Контракт модуля: значение любого признака в строке i зависит исключительно от
values[:i+1] и от календаря. Он не может зависеть от values[i+1:].
Это проверяется структурно в ml/leakage.py, а не декларируется.

Приём, который делает контракт проверяемым: функция row_features получает
срез series.values[:i+1], а не весь ряд. Будущего у неё физически нет.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from ml.calendar_ref import (
    CORRIDOR_HOLIDAYS,
    HOLIDAYS,
    days_since_payday,
    days_since_prev,
    days_to_next,
    days_to_payday,
)
from ml.data import Series

RETURN_LAGS: tuple[int, ...] = (1, 3, 5, 10, 20, 60, 120, 250)
RANGE_WINDOWS: tuple[int, ...] = (30, 90, 180)
VOL_WINDOWS: tuple[int, ...] = (10, 30, 90)
# Прогрев обязан покрывать самое длинное окно признаков, иначе ret_250 первые
# полсотни строк молча отдаёт 0.0 вместо доходности (см. row_features).
WARMUP = max(max(RETURN_LAGS), max(RANGE_WINDOWS), max(VOL_WINDOWS)) + 1


def past_slice(values: np.ndarray, i: int) -> np.ndarray:
    """Единственное место, где ряд режется по времени.

    Вынесено отдельной функцией намеренно: это шов, в который тест на утечку
    подставляет сдвиг среза и проверяет, что проверка действительно ловит
    заглядывание вперёд. Если бы среза не было в одном месте, гарантию
    пришлось бы декларировать, а не проверять.
    """
    return values[: i + 1]


def _bps(new: float, old: float) -> float:
    """Изменение курса в базисных пунктах ДЛЯ ОТПРАВИТЕЛЯ.

    Курс ЦБ = рублей за единицу валюты получателя. Рост курса = валюта дороже =
    отправитель получает меньше. Поэтому знак инвертирован: положительное
    значение всегда означает "лучше для клиента".
    """
    if old <= 0:
        return 0.0
    return -(new - old) / old * 10000.0


def _position_in_range(past: np.ndarray, window: int) -> float:
    """Положение последнего значения в диапазоне min..max окна, 0..100.

    Это НЕ процентиль: чувствительно к одному выбросу на границе окна.
    Настоящий процентиль считает `_share_of_days_beaten` — он нужен для текста
    пуша «выгоднее, чем в 85 % дней», который обещает именно долю дней.
    """
    w = past[-window:]
    lo, hi = float(w.min()), float(w.max())
    if hi <= lo:
        return 50.0
    return (float(w[-1]) - lo) / (hi - lo) * 100.0


def _share_of_days_beaten(past: np.ndarray, window: int) -> float:
    """Доля дней окна, которые сегодняшний курс ПОБИВАЕТ, 0..100 — честный процентиль.

    Ровно та величина, которую обещает разрешённый кейсом текст пуша
    «выгоднее, чем в 85 % дней за последние три месяца»: 85 здесь означает
    «85 % дней были для клиента хуже сегодняшнего».

    Направление здесь легко перепутать, и ошибка была бы не косметической.
    Курс — рубли за единицу валюты получателя, поэтому **низкий курс выгоден
    клиенту**: за те же рубли получатель получает больше. Значит «выгоднее,
    чем в 85 % дней» — это доля дней, когда курс был ВЫШЕ сегодняшнего.
    Доля дней ниже сегодняшнего — величина, обратная нужной: на входе
    `[1, 2, 3, 4, 10]` она равна 100 %, хотя сегодня самый дорогой день окна,
    и пуш «выгоднее, чем в 100 % дней» ушёл бы в худший момент.

    Отличается от `_position_in_range` тем, что не зависит от одного выброса
    на краю окна.
    """
    w = past[-window:]
    if len(w) < 2:
        return 50.0
    return float((w[:-1] > w[-1]).mean()) * 100.0


def _up_share(past: np.ndarray, window: int = 14) -> float:
    w = past[-(window + 1):]
    if len(w) < 2:
        return 50.0
    d = np.diff(w)
    up, dn = float(d[d > 0].sum()), float(-d[d < 0].sum())
    if up + dn == 0:
        return 50.0
    return up / (up + dn) * 100.0


def row_features(
    past: np.ndarray,
    day: dt.date,
    corridor: str,
    gap_days: int,
    ref_past: dict[str, np.ndarray],
    peer_past: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Признаки на дату day. `past` — курсы ПО day включительно, и ничего дальше.

    ref_past — то же для справочных валют (USD, CNY), обрезанное по той же дате.
    """
    v = float(past[-1])
    f: dict[str, float] = {}

    # --- Моментум: доходности на разных горизонтах, в бп для клиента
    for k in RETURN_LAGS:
        f[f"ret_{k}"] = _bps(v, float(past[-1 - k])) if len(past) > k else 0.0

    # --- Серия одинаковых движений (индикатор ТЗ "моментум")
    streak_up = streak_dn = 0
    for j in range(len(past) - 1, 0, -1):
        if past[j] > past[j - 1]:
            if streak_dn:
                break
            streak_up += 1
        elif past[j] < past[j - 1]:
            if streak_up:
                break
            streak_dn += 1
        else:
            break
        if streak_up + streak_dn >= 10:
            break
    f["streak_up"] = float(streak_up)
    f["streak_dn"] = float(streak_dn)

    # --- Уровень: положение в диапазоне (индикатор ТЗ "уровень")
    for w in RANGE_WINDOWS:
        if len(past) >= w:
            f[f"pct_range_{w}"] = _position_in_range(past, w)
            f[f"days_beaten_{w}"] = _share_of_days_beaten(past, w)
            win = past[-w:]
            f[f"dist_min_{w}"] = _bps(v, float(win.min()))
            f[f"dist_max_{w}"] = _bps(v, float(win.max()))
            mu, sd = float(win.mean()), float(win.std())
            f[f"z_{w}"] = (v - mu) / sd if sd > 0 else 0.0
        else:
            f[f"pct_range_{w}"] = f[f"days_beaten_{w}"] = 50.0
            f[f"dist_min_{w}"] = f[f"dist_max_{w}"] = f[f"z_{w}"] = 0.0

    # --- Волатильность
    rets = np.diff(past) / past[:-1] * 10000.0 if len(past) > 1 else np.array([0.0])
    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = float(rets[-w:].std()) if len(rets) >= w else 0.0
    f["vol_ratio"] = f["vol_10"] / f["vol_90"] if f["vol_90"] > 0 else 1.0

    # --- Разворот (индикатор ТЗ "разворот"): сколько назад был экстремум окна
    for w in (30, 90):
        if len(past) >= w:
            win = past[-w:]
            f[f"bars_since_min_{w}"] = float(w - 1 - int(np.argmin(win)))
            f[f"bars_since_max_{w}"] = float(w - 1 - int(np.argmax(win)))
        else:
            f[f"bars_since_min_{w}"] = f[f"bars_since_max_{w}"] = 0.0
    f["up_share_14"] = _up_share(past)

    # --- Кросс-валютные: рубль в целом или именно этот коридор?
    for code, rp in ref_past.items():
        if len(rp) > 5:
            f[f"{code.lower()}_ret_5"] = _bps(float(rp[-1]), float(rp[-6]))
            f[f"{code.lower()}_ret_20"] = (
                _bps(float(rp[-1]), float(rp[-21])) if len(rp) > 20 else 0.0
            )
        else:
            f[f"{code.lower()}_ret_5"] = f[f"{code.lower()}_ret_20"] = 0.0
    # избыточная доходность коридора над долларом — «своё» движение валюты
    f["excess_ret_5"] = f["ret_5"] - f.get("usd_ret_5", 0.0)
    f["excess_ret_20"] = f["ret_20"] - f.get("usd_ret_20", 0.0)

    # --- Календарь (индикатор ТЗ "сезонность") и зарплатное окно
    f["dow"] = float(day.weekday())
    f["dom"] = float(day.day)
    f["month"] = float(day.month)
    f["days_to_payday"] = float(days_to_payday(day))
    f["days_since_payday"] = float(days_since_payday(day))
    f["in_payday_window"] = 1.0 if min(days_to_payday(day), days_since_payday(day)) <= 3 else 0.0
    for name in ("navruz", "eid_adha", "eid_fitr", "new_year", "sep_first"):
        relevant = name in CORRIDOR_HOLIDAYS[corridor]
        f[f"to_{name}"] = float(days_to_next(day, HOLIDAYS[name])) if relevant else 999.0
        f[f"since_{name}"] = float(days_since_prev(day, HOLIDAYS[name])) if relevant else 999.0
    f["to_any_holiday"] = min(
        float(days_to_next(day, HOLIDAYS[n])) for n in CORRIDOR_HOLIDAYS[corridor]
    )
    f["pre_holiday_14d"] = 1.0 if f["to_any_holiday"] <= 14 else 0.0

    # --- Структура ряда публикаций: разрыв до предыдущей публикации
    f["gap_days"] = float(gap_days)
    f["is_after_gap"] = 1.0 if gap_days >= 3 else 0.0

    # --- Отклонение от скользящих средних и режим
    for w in (20, 60):
        if len(past) >= w:
            sma = float(past[-w:].mean())
            f[f"vs_sma_{w}"] = _bps(v, sma)
            # скользящее среднее ровно по w точкам, заканчивающееся на этой же точке
            above = past[-w:] > np.array([past[max(0, len(past) - w + k - w + 1):len(past) - w + k + 1].mean()
                                          for k in range(w)])
            f[f"share_above_sma_{w}"] = float(above.mean())
        else:
            f[f"vs_sma_{w}"] = 0.0
            f[f"share_above_sma_{w}"] = 0.5

    # --- Просадка от локального максимума и ширина диапазона
    for w in (30, 90):
        if len(past) >= w:
            win = past[-w:]
            f[f"drawdown_{w}"] = _bps(v, float(win.max()))
            f[f"range_width_{w}"] = (float(win.max()) - float(win.min())) / float(win.mean()) * 10000.0
        else:
            f[f"drawdown_{w}"] = f[f"range_width_{w}"] = 0.0

    # --- Форма распределения доходностей и ускорение моментума
    if len(rets) >= 30:
        r = rets[-30:]
        sd = float(r.std())
        f["skew_30"] = float(((r - r.mean()) ** 3).mean() / sd**3) if sd > 0 else 0.0
        # sd > 0 обязателен и здесь: на замороженном курсе (жёсткая привязка,
        # залипший фид) все доходности равны нулю, corrcoef возвращает NaN,
        # и дальше падает StandardScaler.
        f["autocorr_30"] = (
            float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 and sd > 0 else 0.0
        )
    else:
        f["skew_30"] = f["autocorr_30"] = 0.0
    f["accel_5_20"] = f["ret_5"] - f["ret_20"]
    f["accel_20_60"] = f["ret_20"] - f["ret_60"]
    f["vol_of_vol"] = (
        float(np.std([rets[-k - 10 : -k].std() for k in range(1, 20)])) if len(rets) >= 30 else 0.0
    )

    # --- Взаимодействие уровня и волатильности: дешёвый день в спокойном рынке
    f["pct_x_vol"] = f["pct_range_90"] * f["vol_30"] / 100.0

    # --- Положение доллара в собственном диапазоне
    usd = ref_past.get("USD")
    if usd is not None and len(usd) >= 90:
        f["usd_pct_range_90"] = _position_in_range(usd, 90)
    else:
        f["usd_pct_range_90"] = 50.0

    # --- Кросс-секция: этот коридор двигается сильнее соседей или это общий рубль?
    if peer_past:
        peer_rets = []
        for arr in peer_past.values():
            if len(arr) > 5:
                peer_rets.append(_bps(float(arr[-1]), float(arr[-6])))
        if peer_rets:
            f["peer_ret_5_mean"] = float(np.mean(peer_rets))
            f["rel_to_peers_5"] = f["ret_5"] - f["peer_ret_5_mean"]
            f["peer_dispersion_5"] = float(np.std(peer_rets))
        else:
            f["peer_ret_5_mean"] = f["rel_to_peers_5"] = f["peer_dispersion_5"] = 0.0
    else:
        f["peer_ret_5_mean"] = f["rel_to_peers_5"] = f["peer_dispersion_5"] = 0.0

    # --- Календарь: положение внутри месяца и квартала
    f["week_of_month"] = float((day.day - 1) // 7)
    f["is_month_end"] = 1.0 if day.day >= 26 else 0.0
    f["is_month_start"] = 1.0 if day.day <= 5 else 0.0
    f["quarter"] = float((day.month - 1) // 3)

    return f


def build_matrix(
    series: dict[str, Series], corridors: tuple[str, ...], refs: tuple[str, ...]
) -> tuple[np.ndarray, list[str], list[tuple[str, int, dt.date]]]:
    """Матрица признаков по всем коридорам (pooled). Возвращает X, имена, индекс строк."""
    # массивы дат неизменны — строим один раз, а не 50 тыс. раз внутри цикла
    ref_dates = {c: np.array(series[c].dates, dtype=object) for c in refs}
    peer_dates = {c: np.array(series[c].dates, dtype=object) for c in corridors}
    rows: list[dict[str, float]] = []
    index: list[tuple[str, int, dt.date]] = []
    for corridor in corridors:
        s = series[corridor]
        for i in range(WARMUP, len(s)):
            day = s.dates[i]
            gap = (day - s.dates[i - 1]).days
            ref_past = {}
            for c in refs:
                # последнее наблюдение справочной валюты НЕ ПОЗЖЕ текущей даты
                j = np.searchsorted(ref_dates[c], day, side="right")
                ref_past[c] = past_slice(series[c].values, j - 1)
            peer_past = {}
            for other in corridors:
                if other == corridor:
                    continue
                k = np.searchsorted(peer_dates[other], day, side="right")
                peer_past[other] = past_slice(series[other].values, k - 1)
            rows.append(
                row_features(past_slice(s.values, i), day, corridor, gap, ref_past, peer_past)
            )
            index.append((corridor, i, day))
    names = sorted(rows[0])
    X = np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
    return X, names, index
