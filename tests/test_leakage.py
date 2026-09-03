"""Тесты честности признаков. Падение любого из них означает дисквалификацию решения."""
from __future__ import annotations

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
    """Очистка обязана падать, если обучение достаёт целью до теста."""
    _, _, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    for h in (1, 5, 20):
        for tr, te, _ in walk_forward_folds(dates, 2021, h):
            assert_no_overlap(dates, tr, te, h)


def test_purge_detects_violation(series):
    """Сама проверка очистки обязана срабатывать на заведомом нарушении."""
    _, _, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    folds = walk_forward_folds(dates, 2021, 5)
    tr, te, _ = folds[0]
    dirty = np.concatenate([tr, te[:5]])  # затащили тестовые строки в обучение
    with pytest.raises(AssertionError):
        assert_no_overlap(dates, dirty, te, 5)


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
    root = Path(__file__).resolve().parent.parent
    for f in SCRIPTS:
        src = (root / f).read_text(encoding="utf-8")
        assert "train_cutoff" in src, f"{f} не берёт порог из обучения"


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

    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _, _, d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    name, report = select_model(X, y, dates, 2021)
    assert name in dict(report)
    # решение не должно зависеть ни от одной строки тестового периода
    X2 = X.copy()
    X2[np.array([d.year >= 2021 for d in dates])] *= 3.0
    name2, _ = select_model(X2, y, dates, 2021)
    assert name == name2, "выбор модели изменился от порчи тестового периода"
