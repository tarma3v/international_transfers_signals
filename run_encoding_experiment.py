"""Ablation: ordinal vs cyclic calendar and ordinal vs one-hot currency.

The production feature matrix uses cyclic calendar coordinates and one-hot
currency flags.  This script reconstructs the previous representation from
the row dates, then compares all four combinations on exactly the same
purged walk-forward folds.

Run:  .venv/bin/python run_encoding_experiment.py
"""
from __future__ import annotations

import calendar
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import REFERENCE_RULE, lift, rate_per_week, reference_rate, train_cutoff
from ml.features import build_matrix
from ml.targets import build_targets
from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds

FIRST_TEST_YEAR = 2021
HORIZON = 5
CYCLIC = tuple(
    f"{part}_{coord}"
    for part in ("dow", "dom", "month", "week_of_month", "quarter")
    for coord in ("sin", "cos")
)


def model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.1, random_state=42)),
    ])


def richer_fourier_calendar(index: list) -> np.ndarray:
    """Более гибкая, но всё ещё непрерывная периодическая кодировка."""
    rows = []
    for _c, _i, day in index:
        cycles = (
            (float(day.weekday()), 7.0, 3),
            (float(day.day - 1), float(calendar.monthrange(day.year, day.month)[1]), 3),
            (float(day.month - 1), 12.0, 3),
            (float((day.day - 1) // 7), 5.0, 2),
            (float((day.month - 1) // 3), 4.0, 2),
        )
        row = []
        for value, period, max_harmonic in cycles:
            for harmonic in range(1, max_harmonic + 1):
                angle = 2.0 * math.pi * harmonic * value / period
                row.extend((math.sin(angle), math.cos(angle)))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def variants(X: np.ndarray, names: list[str], index: list) -> dict[str, np.ndarray]:
    currency_names = tuple(f"currency_{c}" for c in CORRIDORS)
    excluded = set(CYCLIC + currency_names)
    base_cols = [i for i, name in enumerate(names) if name not in excluded]
    cyclic_cols = [names.index(name) for name in CYCLIC]
    currency_cols = [names.index(name) for name in currency_names]

    raw_calendar = np.array([
        [d.weekday(), d.day, d.month, (d.day - 1) // 7, (d.month - 1) // 3]
        for _c, _i, d in index
    ], dtype=float)
    ordinal_currency = np.array(
        [[CORRIDORS.index(c)] for c, _i, _d in index], dtype=float
    )
    base = X[:, base_cols]
    cyclic = X[:, cyclic_cols]
    one_hot = X[:, currency_cols]
    return {
        "старое: числа + currency_id": np.column_stack([base, raw_calendar, ordinal_currency]),
        "только cyclic calendar": np.column_stack([base, cyclic, ordinal_currency]),
        "только one-hot currency": np.column_stack([base, raw_calendar, one_hot]),
        "новое: cyclic + one-hot": np.column_stack([base, cyclic, one_hot]),
        "raw + cyclic + one-hot": np.column_stack([base, raw_calendar, cyclic, one_hot]),
        "Fourier 2–3 гармоники + one-hot": np.column_stack(
            [base, richer_fourier_calendar(index), one_hot]
        ),
    }


def main() -> None:
    series = load()
    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _c, _i, d in index], dtype=object)
    y = build_targets(series, index)[f"fav_h{HORIZON}"]
    reach = target_reach_dates(index, series, HORIZON)
    folds = walk_forward_folds(dates, FIRST_TEST_YEAR, HORIZON, reach=reach)
    rate = reference_rate(
        BASELINES[REFERENCE_RULE](X, names), dates, FIRST_TEST_YEAR
    )

    print("Цель: текущий курс не будет побит в следующие 5 публикаций")
    print("Проверка: purged walk-forward, тестовые годы 2021–2026, одна логрегрессия")
    print(f"Рабочая частота взята до теста: {rate * 100:.1f}%\n")
    print(f"{'вариант':<31}{'фич':>6}{'AUC':>9}{'Brier↓':>10}{'lift':>8}{'сигн/нед':>11}")

    predictions: dict[str, np.ndarray] = {}
    oos = np.zeros(len(y), dtype=bool)
    for label, Xm in variants(X, names, index).items():
        score = np.full(len(y), np.nan)
        fired = np.zeros(len(y), dtype=bool)
        for tr_i, te_i, _year in folds:
            assert_no_overlap(
                dates, tr_i, te_i, HORIZON, index=index, series=series
            )
            tr = tr_i[~np.isnan(y[tr_i])]
            te = te_i[~np.isnan(y[te_i])]
            if len(tr) < 400 or len(te) < 30:
                continue
            fitted = model().fit(Xm[tr], y[tr])
            train_score = fitted.predict_proba(Xm[tr])[:, 1]
            score[te] = fitted.predict_proba(Xm[te])[:, 1]
            fired[te] = score[te] >= train_cutoff(train_score, rate)
            oos[te] = True

        valid = oos & ~np.isnan(score) & ~np.isnan(y)
        auc = roc_auc_score(y[valid], score[valid])
        brier = brier_score_loss(y[valid], score[valid])
        lf, _hit, n = lift(fired, y, scope=valid)
        weekly = rate_per_week(n, len(CORRIDORS), dates, valid)
        predictions[label] = score
        print(f"{label:<31}{Xm.shape[1]:>6}{auc:>9.4f}{brier:>10.4f}{lf:>8.3f}{weekly:>11.2f}")

    print("\nAUC по тестовым годам (старое → новое):")
    old = predictions["старое: числа + currency_id"]
    new = predictions["новое: cyclic + one-hot"]
    for year in sorted({d.year for d in dates[oos]}):
        mask = oos & np.array([d.year == year for d in dates]) & ~np.isnan(y)
        a_old = roc_auc_score(y[mask], old[mask])
        a_new = roc_auc_score(y[mask], new[mask])
        print(f"  {year}: {a_old:.4f} → {a_new:.4f}  ({a_new - a_old:+.4f})")


if __name__ == "__main__":
    main()
