"""Two-stage policy: keep the strong range-tail rule and model only extra signals.

The model family is selected on an inner split entirely before 2021.  In every
walk-forward fold the rescue cutoff is computed from that fold's training
scores, never from the test year.  This tests whether ML can raise the signal
frequency to the product band without diluting the strong tail too much.

Run:  .venv/bin/python run_tail_rescue_experiment.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from sklearn.metrics import roc_auc_score

from ml.data import CORRIDORS, REFERENCE, load
from ml.evaluate import lift, rate_per_week
from ml.features import build_matrix
from ml.models import make_classifiers
from ml.targets import build_targets
from ml.validation import target_reach_dates, walk_forward_folds

FIRST_TEST_YEAR = 2021
HORIZON = 5
TAIL_THRESHOLD = 95.0
TARGET_RATES = (0.20, 0.25)  # approximately 1.0 and 1.25 signals/corridor/week


def choose_rescue_model(X, y, dates, reach, tail) -> tuple[str, list[tuple[str, float]]]:
    """Choose the conditional ranker without looking at any 2021+ observation."""
    dev = np.array([d < dt.date(FIRST_TEST_YEAR, 1, 1) for d in dates]) & ~np.isnan(y)
    wall = max(dates[dev]) - dt.timedelta(days=180)
    train = dev & np.array([d <= wall and r <= wall for d, r in zip(dates, reach)]) & ~tail
    valid = dev & np.array([d > wall for d in dates]) & ~tail
    report = []
    for name, model in make_classifiers().items():
        model.fit(X[train], y[train])
        score = model.predict_proba(X[valid])[:, 1]
        report.append((name, float(roc_auc_score(y[valid], score))))
    report.sort(key=lambda row: -row[1])
    return report[0][0], report


def main() -> None:
    series = load()
    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    targets = build_targets(series, index)
    y = targets[f"fav_h{HORIZON}"]
    dates = np.array([d for _c, _i, d in index], dtype=object)
    reach = target_reach_dates(index, series, HORIZON)
    tail = X[:, names.index("pct_range_90")] >= TAIL_THRESHOLD

    selected, selection = choose_rescue_model(X, y, dates, reach, tail)
    print("ВЫБОР RESCUE-МОДЕЛИ ТОЛЬКО ДО 2021")
    for name, auc in selection:
        print(f"  {name:<25} AUC={auc:.3f}")
    print(f"Выбрана: {selected}\n")

    model_signal = {rate: np.zeros(len(y), dtype=bool) for rate in TARGET_RATES}
    model_only_score = np.full(len(y), np.nan)
    test_scope = np.zeros(len(y), dtype=bool)
    for train_idx, test_idx, year in walk_forward_folds(
        dates, FIRST_TEST_YEAR, HORIZON, reach=reach
    ):
        train_idx = train_idx[~np.isnan(y[train_idx])]
        test_idx = test_idx[~np.isnan(y[test_idx])]
        rescue_train = train_idx[~tail[train_idx]]
        model = make_classifiers()[selected]
        model.fit(X[rescue_train], y[rescue_train])
        train_score = model.predict_proba(X[rescue_train])[:, 1]
        test_score = model.predict_proba(X[test_idx])[:, 1]
        model_only_score[test_idx] = test_score
        test_scope[test_idx] = True

        observed_tail_rate = float(tail[train_idx].mean())
        for target_rate in TARGET_RATES:
            conditional_rate = np.clip(
                (target_rate - observed_tail_rate) / max(1e-12, 1.0 - observed_tail_rate),
                0.0,
                1.0,
            )
            cutoff = float(np.quantile(train_score, 1.0 - conditional_rate))
            model_signal[target_rate][test_idx] = tail[test_idx] | (
                (~tail[test_idx]) & (test_score >= cutoff)
            )
        print(
            f"{year}: train={len(train_idx)}, tail={observed_tail_rate:.3f}, "
            f"test={len(test_idx)}"
        )

    print("\nЧЕСТНЫЙ OOS 2021–2026, h=5")
    print(f"{'политика':<31}{'сигн/нед':>11}{'lift':>9}{'hit-rate':>11}{'n':>7}")
    candidates = [("только pct_range_90 >= 95", tail)] + [
        (f"tail + rescue до {rate:.0%}", fired) for rate, fired in model_signal.items()
    ]
    for label, fired in candidates:
        lf, _base, n = lift(fired, y, scope=test_scope)
        active = fired & test_scope
        frequency = rate_per_week(n, len(CORRIDORS), dates, test_scope)
        hit = float(y[active].mean())
        print(f"{label:<31}{frequency:>11.2f}{lf:>9.3f}{hit:>11.3f}{n:>7}")

    valid_score = test_scope & ~np.isnan(model_only_score)
    print(f"\nAUC rescue-score на всех OOS строках: "
          f"{roc_auc_score(y[valid_score], model_only_score[valid_score]):.3f}")


if __name__ == "__main__":
    main()
