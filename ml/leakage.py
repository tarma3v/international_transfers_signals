"""Структурная проверка отсутствия заглядывания в будущее.

Принцип: если признак в строке i зависит от значений после i, то порча будущего
изменит этот признак. Портим будущее и требуем побитового совпадения.

Тест на самом тесте: намеренно сломанный признак ОБЯЗАН быть пойман. Проверка,
не умеющая находить утечку, ничего не гарантирует.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS, REFERENCE, Series, load
from ml.features import build_matrix


def corrupt_future(series: dict[str, Series], cut: dt.date, factor: float = 3.0) -> dict[str, Series]:
    """Копия данных, где всё СТРОГО ПОСЛЕ даты cut умножено на factor."""
    out: dict[str, Series] = {}
    for code, s in series.items():
        v = s.values.copy()
        mask = np.array([d > cut for d in s.dates])
        v[mask] *= factor
        out[code] = Series(code, s.dates.copy(), v)
    return out


def check_no_lookahead(
    series: dict[str, Series], cut: dt.date, factor: float = 3.0
) -> tuple[bool, list[str], int]:
    """Возвращает (чисто, список протёкших признаков, сколько строк проверено)."""
    X_ref, names, idx = build_matrix(series, CORRIDORS, REFERENCE)
    X_bad, names_bad, idx_bad = build_matrix(corrupt_future(series, cut, factor), CORRIDORS, REFERENCE)
    assert names == names_bad and idx == idx_bad, "структура матрицы изменилась"

    past_rows = np.array([d <= cut for _, _, d in idx])
    diff = ~np.isclose(X_ref[past_rows], X_bad[past_rows], rtol=1e-12, atol=1e-12, equal_nan=True)
    leaked = [names[j] for j in range(len(names)) if diff[:, j].any()]
    return (not leaked), leaked, int(past_rows.sum())


def check_detector_works(series: dict[str, Series], cut: dt.date) -> bool:
    """Проверка самой проверки.

    Подставляем в шов past_slice реальную ошибку — срез, захватывающий 5 дней
    вперёд. Именно так выглядит настоящая утечка: не злой умысел, а сдвиг
    индекса. Детектор ОБЯЗАН её поймать, иначе его вердикт ничего не стоит.
    """
    import ml.features as F

    original = F.past_slice

    def leaky(values, i):
        return values[: i + 6]

    F.past_slice = leaky
    try:
        clean, leaked, _ = check_no_lookahead(series, cut)
    finally:
        F.past_slice = original
    return not clean


def check_purge(train_idx: np.ndarray, test_start: int, horizon: int) -> bool:
    """Ни одна обучающая строка не должна смотреть целевой переменной в тест."""
    return bool(np.all(train_idx + horizon < test_start))


if __name__ == "__main__":
    s = load()
    cut = dt.date(2023, 6, 30)

    print("=" * 78)
    print("ТЕСТ 1. Проверка умеет ловить утечку (подсовываем заглядывание на 5 дней)")
    ok_detector = check_detector_works(s, cut)
    print(f"   ловушка поймана: {'ДА' if ok_detector else 'НЕТ — тест бесполезен'}")

    print("\nТЕСТ 2. Порча будущего не меняет ни одного признака в прошлом")
    clean, leaked, n = check_no_lookahead(s, cut)
    print(f"   срез: {cut} | проверено строк: {n}")
    print(f"   протёкших признаков: {len(leaked)}{' -> ' + ', '.join(leaked) if leaked else ''}")

    print("\nТЕСТ 3. То же на другой дате среза и другом множителе")
    clean2, leaked2, n2 = check_no_lookahead(s, dt.date(2021, 3, 15), factor=0.2)
    print(f"   срез: 2021-03-15 | проверено строк: {n2} | протёкших: {len(leaked2)}")

    print("\n" + "=" * 78)
    verdict = ok_detector and clean and clean2
    print(f"ИТОГ: {'ЗАГЛЯДЫВАНИЯ В БУДУЩЕЕ НЕТ' if verdict else 'ОБНАРУЖЕНА УТЕЧКА'}")
    raise SystemExit(0 if verdict else 1)
