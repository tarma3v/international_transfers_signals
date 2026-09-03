"""Тесты честности признаков. Падение любого из них означает дисквалификацию решения."""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from ml.data import CORRIDORS, REFERENCE, load
from ml.features import build_matrix, past_slice
from ml.leakage import check_detector_works, check_no_lookahead
from ml.targets import benefit_forward_only, target_now_favourable
from ml.validation import assert_no_overlap, walk_forward_folds


@pytest.fixture(scope="module")
def series():
    return load()


def test_detector_catches_planted_leak(series):
    """Проверка обязана ловить утечку, иначе её вердикт ничего не значит."""
    assert check_detector_works(series, dt.date(2023, 6, 30))


@pytest.mark.parametrize(
    "cut,factor",
    [(dt.date(2023, 6, 30), 3.0), (dt.date(2021, 3, 15), 0.2), (dt.date(2024, 9, 1), 10.0)],
)
def test_no_lookahead(series, cut, factor):
    """Порча будущего не должна менять ни одного признака в прошлом."""
    clean, leaked, n = check_no_lookahead(series, cut, factor)
    assert clean, f"признаки смотрят в будущее: {leaked}"
    assert n > 500


def test_past_slice_excludes_present_future():
    v = np.arange(10.0)
    assert past_slice(v, 4).tolist() == [0, 1, 2, 3, 4]
    assert len(past_slice(v, 0)) == 1


def test_purge_blocks_overlap(series):
    """Очистка обязана падать, если обучение достаёт целью до теста.

    Проверяется РАБОЧАЯ проводка — та, которой пользуются все скрипты: фолды
    строятся по точным датам достижения цели, а сторож пересчитывает эти даты
    сам. Вызов обеих функций без `reach`/`index` проверял бы календарную
    ветку с запасом h + 10 дней, которую не использует ни один скрипт и
    которая на h = 20 пропускает настоящую утечку (это отдельно показано
    в `test_calendar_guard_is_blind_to_real_overlap`).
    """
    from ml.validation import target_reach_dates

    _, _, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    for h in (1, 5, 20):
        reach = target_reach_dates(index, series, h)
        for tr, te, _ in walk_forward_folds(dates, 2021, h, reach=reach):
            assert_no_overlap(dates, tr, te, h, index=index, series=series)


def test_calendar_guard_is_blind_to_real_overlap(series):
    """Почему сторож обязан пересчитывать даты, а не сравнивать календарь.

    Календарная ветка сторожа сравнивает «последняя дата обучения + h дней»
    с началом теста. На h = 20 существует фолд, где обучающие строки физически
    достают целью внутрь теста, а такое сравнение всё равно молчит: 20
    публикаций занимают до 40 календарных дней. Точная ветка на том же фолде
    падает. Тест фиксирует оба факта, чтобы «зелёный» прогон нельзя было
    получить, вызвав слабую ветку.
    """
    from ml.validation import target_reach_dates

    _, _, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    h = 20
    reach = target_reach_dates(index, series, h)
    tr, te = _legacy_calendar_fold(dates, index, 2021, h)
    first_test = min(dates[i] for i in te)
    assert any(reach[i] >= first_test for i in tr), "фолд обязан протекать"

    assert_no_overlap(dates, tr, te, h)  # календарный сторож молчит
    with pytest.raises(AssertionError):
        assert_no_overlap(dates, tr, te, h, index=index, series=series)


def test_purge_detects_violation(series):
    """Сама проверка очистки обязана срабатывать на заведомом нарушении."""
    _, _, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    folds = walk_forward_folds(dates, 2021, 5)
    tr, te, _ = folds[0]
    dirty = np.concatenate([tr, te[:5]])  # затащили тестовые строки в обучение
    with pytest.raises(AssertionError):
        assert_no_overlap(dates, dirty, te, 5, index=index, series=series)


def test_target_is_strictly_forward():
    """Цель зависит только от будущего и от текущего значения, но не от прошлого."""
    v = np.array([10.0, 9.0, 8.0, 11.0, 12.0, 13.0])
    before = target_now_favourable(v, 2, 3)
    v2 = v.copy()
    v2[0] = 999.0  # меняем прошлое
    assert target_now_favourable(v2, 2, 3) == before


def test_forward_benefit_ignores_past():
    v = np.array([10.0, 9.0, 8.0, 11.0, 12.0, 13.0])
    a = benefit_forward_only(v, 2, 3)
    v2 = v.copy()
    v2[:2] = 500.0
    assert benefit_forward_only(v2, 2, 3) == a


def test_benefit_sign_is_client_oriented():
    """Курс ЦБ растёт = валюта дороже = клиенту хуже = выгода отрицательна."""
    rising = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    assert benefit_forward_only(rising, 0, 3) > 0  # сегодня дешевле будущего
    falling = np.array([14.0, 13.0, 12.0, 11.0, 10.0])
    assert benefit_forward_only(falling, 0, 3) < 0


def test_no_nan_in_feature_matrix(series):
    X, names, _ = build_matrix(series, CORRIDORS, REFERENCE)
    assert np.all(np.isfinite(X)), "в матрице признаков есть NaN или inf"
    assert len(names) == X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Рабочая точка. Признаки могут быть чистыми, а решение «когда срабатывать» —
# взято из теста. Ворота на утечку такое не ловят по конструкции: они смотрят
# на матрицу признаков. Эти тесты закрывают именно эту дыру.
# ─────────────────────────────────────────────────────────────────────────────

SCRIPTS = [
    "run_experiment.py", "run_boosting.py", "run_two_models.py",
    "summarize.py", "check_stability.py", "validate_result.py", "make_figures.py",
]


def test_no_operating_point_from_test_scores():
    """Порог нельзя брать квантилем по тестовым оценкам — такой функции нет."""
    import ml.evaluate as ev

    assert not hasattr(ev, "threshold_at_rate"), (
        "порог как квантиль по OOS-оценкам — это подгонка рабочей точки под тест; "
        "такой функции в проекте быть не должно"
    )
    assert hasattr(ev, "train_cutoff")
    root = Path(__file__).resolve().parent.parent
    for f in SCRIPTS + ["ml/evaluate.py"]:
        assert "threshold_at_rate" not in (root / f).read_text(encoding="utf-8"), f


def test_every_scoring_script_takes_cutoff_from_training():
    """Слабая проверка: имя функции присутствует. Настоящую делает тест ниже."""
    root = Path(__file__).resolve().parent.parent
    for f in SCRIPTS:
        src = (root / f).read_text(encoding="utf-8")
        assert "train_cutoff" in src, f"{f} не берёт порог из обучения"


TEST_MASKS = {"te", "te_i", "teA", "teB", "test", "test_idx", "oos", "is_test"}


def _quantiles_over_test_rows(src: str) -> list[tuple[int, str]]:
    """Места, где квантиль берётся от массива, суженного тестовой маской.

    Проверка по существу, а не по имени: запрещён ПРИЁМ «порог как квантиль по
    оценкам теста», а не конкретная функция. Поиск подстроки `train_cutoff`
    его не ловит — импорт и комментарий остаются на месте, даже если реальный
    вызов подменить на `np.quantile(score[te], ...)`.
    """
    bad: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in {"quantile", "percentile", "nanquantile", "nanpercentile"}:
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.Subscript):
            continue
        used = {n.id for n in ast.walk(arg.slice) if isinstance(n, ast.Name)}
        if used & TEST_MASKS:
            bad.append((node.lineno, ", ".join(sorted(used & TEST_MASKS))))
    return bad


def test_no_script_computes_a_quantile_over_test_rows():
    """Рабочая точка не может браться с теста — проверяется разбором кода.

    Падает, если в любом скрипте появится `np.quantile(что-то[тестовая маска])`:
    именно так выглядит подгонка порога под тест, и именно её не видит ни
    проверка на утечку в признаках, ни поиск по имени функции.
    """
    root = Path(__file__).resolve().parent.parent
    files = SCRIPTS + ["run_product_numbers.py"] + [
        str(q.relative_to(root)) for q in sorted((root / "ml").glob("*.py"))
    ]
    offenders = {
        f: bad
        for f in files
        if (bad := _quantiles_over_test_rows((root / f).read_text(encoding="utf-8")))
    }
    assert not offenders, f"квантиль по тестовым строкам: {offenders}"


def test_train_cutoff_is_empirical_quantile_of_training_only():
    from ml.evaluate import train_cutoff

    rng = np.random.default_rng(0)
    tr = rng.normal(size=5000)
    cut = train_cutoff(tr, 0.19)
    assert abs(cut - float(np.quantile(tr, 0.81))) < 1e-12
    # доля срабатываний на обучении = целевая
    assert abs(float((tr >= cut).mean()) - 0.19) < 0.01


def test_honest_cutoff_lets_test_frequency_drift():
    """Главное следствие честного порога: частота на тесте НЕ равна целевой.

    Ровная частота — признак того, что порог подогнали под тест.
    """
    from ml.evaluate import train_cutoff

    rng = np.random.default_rng(1)
    tr = rng.normal(size=4000)
    te = rng.normal(loc=0.5, size=1000)  # тестовый период сдвинут
    cut = train_cutoff(tr, 0.19)
    assert float((te >= cut).mean()) > 0.30, "порог обязан быть слеп к тесту"


def test_model_choice_is_made_before_the_test(series):
    """Выбор модели по её результату на тесте — то же заглядывание уровнем выше."""
    from ml.selection import select_model
    from ml.targets import build_targets
    from ml.validation import target_reach_dates

    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    # горизонт и точные даты достижения передаём так же, как это делают скрипты:
    # без них внутренний сплит режется нулевым зазором, и тест проверял бы
    # проводку, которой в проекте нет
    reach = target_reach_dates(index, series, 5)
    kw = {"horizon": 5, "reach": reach}
    name, report = select_model(X, y, dates, 2021, **kw)
    assert name in dict(report)
    # решение не должно зависеть ни от одной строки тестового периода
    X2 = X.copy()
    X2[np.array([d.year >= 2021 for d in dates])] *= 3.0
    name2, _ = select_model(X2, y, dates, 2021, **kw)
    assert name == name2, "выбор модели изменился от порчи тестового периода"


# ─────────────────────────────────────────────────────────────────────────────
# Единицы измерения и корректность метрик. Проверка, которая не может упасть,
# ничего не гарантирует, поэтому каждый тест ниже сначала показан падающим на
# старом поведении и только потом зафиксирован.
# ─────────────────────────────────────────────────────────────────────────────


def _legacy_calendar_fold(dates, index, first_test_year, horizon):
    """Фолд по прежней арифметике: запас отсчитан в КАЛЕНДАРНЫХ днях.

    Воспроизводится намеренно: тесты ниже обязаны показать, что именно эта
    арифметика протекает, иначе они ничего не пинят.
    """
    del index
    test_start = dt.date(first_test_year, 1, 1)
    test_end = dt.date(first_test_year, 12, 31)
    cutoff = test_start - dt.timedelta(days=horizon + 4)
    is_test = np.array([test_start <= d <= test_end for d in dates])
    is_train = np.array([d <= cutoff for d in dates])
    return np.where(is_train)[0], np.where(is_test)[0]


def test_purge_measures_horizon_in_publications(series):
    """h публикаций занимают больше h календарных дней — и это протекает."""
    from ml.validation import target_reach_dates, walk_forward_folds

    _X, _names, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    h = 20
    reach = target_reach_dates(index, series, h)

    tr_old, te_old = _legacy_calendar_fold(dates, index, 2021, h)
    first_test = min(dates[i] for i in te_old)
    leaked = sum(1 for i in tr_old if reach[i] >= first_test)
    assert leaked > 0, "старая арифметика обязана протекать — иначе тест пустой"

    for tr, te, _ in walk_forward_folds(dates, 2021, h, reach=reach):
        ft = min(dates[i] for i in te)
        assert all(reach[i] < ft for i in tr), "точная очистка сама протекла"


def test_assert_no_overlap_can_actually_fail(series):
    """Сторож обязан падать на фолде, собранном по прежней арифметике."""
    import pytest as _pytest

    from ml.validation import assert_no_overlap

    _X, _n, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    h = 20
    tr_old, te_old = _legacy_calendar_fold(dates, index, 2021, h)
    with _pytest.raises(AssertionError):
        assert_no_overlap(dates, tr_old, te_old, h, index=index, series=series)


def test_lift_denominator_follows_scope():
    """База считается по тем же строкам, на которых оценивается сигнал."""
    from ml.evaluate import lift

    y = np.array([1.0] * 50 + [0.0] * 50 + [1.0] * 10 + [0.0] * 90)
    scope = np.array([False] * 100 + [True] * 100)
    fired = np.zeros(200, bool)
    fired[100:120] = True          # 10 попаданий из 20 внутри scope
    lf_scoped, base_scoped, _ = lift(fired, y, scope=scope)
    lf_all, base_all, _ = lift(fired, y)
    assert abs(base_scoped - 0.10) < 1e-9
    assert abs(base_all - 0.30) < 1e-9
    assert lf_scoped > lf_all, "неверная база меняет lift — этот тест ловит подмену"


def test_clustered_ci_is_wider_when_rows_are_correlated():
    """Построчный бутстрап на коррелированных строках занижает интервал."""
    from ml.evaluate import bootstrap_ci

    rng = np.random.default_rng(7)
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=int(k)) for k in range(120)]
    vals, ds = [], []
    for d in days:
        shock = rng.normal(0, 10.0)        # общий для всех коридоров в этот день
        for _ in range(5):
            vals.append(shock + rng.normal(0, 1.0))
            ds.append(d)
    x = np.array(vals)
    lo_i, hi_i = bootstrap_ci(x)
    lo_c, hi_c = bootstrap_ci(x, dates=np.array(ds, dtype=object))
    assert (hi_c - lo_c) > 1.5 * (hi_i - lo_i), (
        "кластеризованный интервал обязан быть заметно шире построчного"
    )


def test_corruptions_cover_sign_leaks(series):
    """Масштабирующая порча слепа к знакам — вторая порча обязана их видеть."""
    from ml.leakage import corruptions_are_complementary

    assert corruptions_are_complementary(series, dt.date(2023, 6, 30))


def test_features_survive_a_frozen_rate():
    """Замороженный курс (жёсткая привязка, залипший фид) не даёт NaN."""
    from ml.features import row_features

    frozen = np.full(300, 12.5)
    f = row_features(frozen, dt.date(2024, 6, 3), "TJS", 1, {"USD": frozen})
    bad = [k for k, v in f.items() if not np.isfinite(v)]
    assert not bad, f"нефинитные признаки на замороженном ряду: {bad}"
