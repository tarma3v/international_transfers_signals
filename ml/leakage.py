"""Структурная проверка отсутствия заглядывания в будущее.

Принцип: если признак в строке i зависит от значений после i, то порча будущего
изменит этот признак. Портим будущее и требуем побитового совпадения.

Порча делается ДВУМЯ разными способами, и это не перестраховка. Умножение на
константу сохраняет знаки и порядок приращений, поэтому признак, читающий
направление будущего (например, «сколько падений в следующие пять публикаций»),
такую порчу переживает без изменений и объявляется чистым. Перестановка будущего
ломает ранги и знаки, но сохраняет множество значений. Утечка обязана быть
поймана хотя бы одной из порч, а тест на самом тесте требует, чтобы детектор
ловил обе подставные утечки — и на величинах, и на знаках.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS, REFERENCE, Series, load
from ml.features import build_matrix


def corrupt_future(
    series: dict[str, Series], cut: dt.date, factor: float = 3.0, mode: str = "scale"
) -> dict[str, Series]:
    """Копия данных, где всё СТРОГО ПОСЛЕ даты cut испорчено.

    mode="scale"   — умножение на factor: ломает величины, сохраняет знаки и ранги.
    mode="shuffle" — перестановка будущего: ломает знаки и ранги, сохраняет величины.
    Признак, зависящий от будущего, обязан измениться хотя бы от одной из порч.
    """
    rng = np.random.default_rng(20260903)
    out: dict[str, Series] = {}
    for code, s in series.items():
        v = s.values.copy()
        mask = np.array([d > cut for d in s.dates])
        if mode == "scale":
            v[mask] *= factor
        elif mode == "shuffle":
            fut = v[mask]
            if len(fut) > 1:
                perm = rng.permutation(len(fut))
                # гарантируем, что порядок реально изменился
                if np.array_equal(perm, np.arange(len(fut))):
                    perm = perm[::-1]
                v[mask] = fut[perm]
        else:
            raise ValueError(f"неизвестный режим порчи: {mode}")
        out[code] = Series(code, s.dates.copy(), v)
    return out


def check_no_lookahead(
    series: dict[str, Series], cut: dt.date, factor: float = 3.0, mode: str = "both"
) -> tuple[bool, list[str], int]:
    """Возвращает (чисто, список протёкших признаков, сколько строк проверено).

    mode="both" прогоняет обе порчи и объединяет находки: утечка на знаках
    видна только перестановкой, утечка на величинах — только масштабированием.
    """
    modes = ("scale", "shuffle") if mode == "both" else (mode,)
    X_ref, names, idx = build_matrix(series, CORRIDORS, REFERENCE)
    past_rows = np.array([d <= cut for _, _, d in idx])
    leaked: list[str] = []
    for m in modes:
        X_bad, names_bad, idx_bad = build_matrix(
            corrupt_future(series, cut, factor, m), CORRIDORS, REFERENCE
        )
        assert names == names_bad and idx == idx_bad, "структура матрицы изменилась"
        diff = ~np.isclose(
            X_ref[past_rows], X_bad[past_rows], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        for j in range(len(names)):
            if diff[:, j].any() and names[j] not in leaked:
                leaked.append(names[j])
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
        clean, _leaked, _ = check_no_lookahead(series, cut)
    finally:
        F.past_slice = original
    return not clean


def corruptions_are_complementary(series: dict[str, Series], cut: dt.date) -> bool:
    """Две порчи ловят разные классы утечек — иначе вторая бессмысленна.

    Берём две статистики будущего: «средняя величина» и «сколько падений подряд».
    Масштабирующая порча обязана изменить первую и НЕ изменить вторую (умножение
    на положительную константу сохраняет знаки приращений). Перестановка обязана
    изменить вторую. Если это не так, набор порч не покрывает утечки на знаках.
    """
    code = next(iter(series))
    v = series[code].values
    fut_mask = np.array([d > cut for d in series[code].dates])
    scaled = corrupt_future(series, cut, 3.0, "scale")[code].values
    shuffled = corrupt_future(series, cut, 3.0, "shuffle")[code].values

    def magnitude(x):
        return float(x[fut_mask].mean())

    def signs(x):
        return int((np.diff(x[fut_mask]) < 0).sum())

    scale_blind_to_signs = signs(scaled) == signs(v)
    scale_sees_magnitude = magnitude(scaled) != magnitude(v)
    shuffle_sees_signs = signs(shuffled) != signs(v)
    return bool(scale_sees_magnitude and scale_blind_to_signs and shuffle_sees_signs)


if __name__ == "__main__":
    s = load()
    cut = dt.date(2023, 6, 30)

    print("=" * 78)
    print("ТЕСТ 0. Две порчи ловят разные классы утечек")
    ok_compl = corruptions_are_complementary(s, cut)
    print(f"   масштаб слеп к знакам, перестановка их видит: {'ДА' if ok_compl else 'НЕТ'}")

    print("\nТЕСТ 1. Проверка умеет ловить утечку (подсовываем заглядывание на 5 дней)")
    ok_detector = check_detector_works(s, cut)
    print(f"   ловушка поймана: {'ДА' if ok_detector else 'НЕТ — тест бесполезен'}")

    print("\nТЕСТ 2. Две порчи будущего (масштаб и перестановка) не меняют прошлого")
    clean, leaked, n = check_no_lookahead(s, cut)
    print(f"   срез: {cut} | проверено строк: {n}")
    print(f"   протёкших признаков: {len(leaked)}{' -> ' + ', '.join(leaked) if leaked else ''}")

    print("\nТЕСТ 3. То же на другой дате среза и другом множителе")
    clean2, leaked2, n2 = check_no_lookahead(s, dt.date(2021, 3, 15), factor=0.2)
    print(f"   срез: 2021-03-15 | проверено строк: {n2} | протёкших: {len(leaked2)}")

    print("\n" + "=" * 78)
    verdict = ok_compl and ok_detector and clean and clean2
    print(f"ИТОГ: {'ЗАГЛЯДЫВАНИЯ В БУДУЩЕЕ НЕТ' if verdict else 'ОБНАРУЖЕНА УТЕЧКА'}")
    raise SystemExit(0 if verdict else 1)
