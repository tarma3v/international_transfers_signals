"""Seasonality by currency, separated into exploration and honest prediction.

Run:  .venv/bin/python analyze_currency_seasonality.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS, REFERENCE, load
from ml.features import build_matrix
from ml.targets import build_targets
from ml.validation import target_reach_dates, walk_forward_folds

FIRST_TEST_YEAR = 2021
HORIZON = 5


def is_calendar_feature(name: str) -> bool:
    exact = {
        "days_to_payday", "days_since_payday", "in_payday_window",
        "to_any_holiday", "pre_holiday_14d", "is_month_end", "is_month_start",
    }
    prefixes = (
        "dow_", "dom_", "month_", "week_of_month_", "quarter_", "to_", "since_",
    )
    return name in exact or name.startswith(prefixes)


def make_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.1, random_state=42)),
    ])


def main() -> None:
    series = load()
    X, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    dates = np.array([d for _c, _i, d in index], dtype=object)
    row_currency = np.array([c for c, _i, _d in index], dtype=object)
    y = build_targets(series, index)[f"fav_h{HORIZON}"]
    reach = target_reach_dates(index, series, HORIZON)
    folds = walk_forward_folds(dates, FIRST_TEST_YEAR, HORIZON, reach=reach)

    calendar_cols = [i for i, n in enumerate(names) if is_calendar_feature(n)]
    market_cols = [
        i for i, n in enumerate(names)
        if not is_calendar_feature(n) and not n.startswith("currency_")
    ]
    variants = {
        "season": calendar_cols,
        "market": market_cols,
        "market+season": market_cols + calendar_cols,
    }
    scores = {
        c: {label: np.full(len(y), np.nan) for label in variants}
        for c in CORRIDORS
    }

    for train_idx, test_idx, _year in folds:
        for currency in CORRIDORS:
            is_currency = row_currency == currency
            tr = train_idx[is_currency[train_idx] & ~np.isnan(y[train_idx])]
            te = test_idx[is_currency[test_idx] & ~np.isnan(y[test_idx])]
            if len(tr) < 150 or len(te) < 30 or len(np.unique(y[tr])) < 2:
                continue
            for label, cols in variants.items():
                fitted = make_model().fit(X[tr][:, cols], y[tr])
                scores[currency][label][te] = fitted.predict_proba(X[te][:, cols])[:, 1]

    print("ЧЕСТНАЯ ПРОВЕРКА: отдельная модель на каждый коридор, h=5")
    print("Каждый тестовый год предсказывается только по предыдущим годам.")
    print(f"Календарных признаков: {len(calendar_cols)}, рыночных: {len(market_cols)}\n")
    print(f"{'валюта':<9}{'base':>8}{'season AUC':>13}{'market AUC':>13}"
          f"{'full AUC':>11}{'Δ season':>11}{'лет лучше':>12}")
    for currency in CORRIDORS:
        valid = (row_currency == currency) & ~np.isnan(scores[currency]["market+season"]) & ~np.isnan(y)
        base = float(y[valid].mean())
        aucs = {
            label: roc_auc_score(y[valid], score[valid])
            for label, score in scores[currency].items()
        }
        better = total = 0
        for year in sorted({d.year for d in dates[valid]}):
            m = valid & np.array([d.year == year for d in dates])
            if len(np.unique(y[m])) < 2:
                continue
            total += 1
            a_market = roc_auc_score(y[m], scores[currency]["market"][m])
            a_full = roc_auc_score(y[m], scores[currency]["market+season"][m])
            better += a_full > a_market
        print(f"{currency:<9}{base:>8.3f}{aucs['season']:>13.3f}{aucs['market']:>13.3f}"
              f"{aucs['market+season']:>11.3f}{aucs['market+season']-aucs['market']:>+11.3f}"
              f"{f'{better}/{total}':>12}")

    print("\nРАЗВЕДОЧНО: самые сильные месяцы на тестовом периоде 2021–2026")
    print("Это описание уже увиденных данных, а не честный прогноз и не выбор фичи.")
    test_period = np.array([d.year >= FIRST_TEST_YEAR for d in dates]) & ~np.isnan(y)
    for currency in CORRIDORS:
        cmask = test_period & (row_currency == currency)
        base = float(y[cmask].mean())
        rows = []
        for month in range(1, 13):
            m = cmask & np.array([d.month == month for d in dates])
            if m.sum() < 20:
                continue
            stable = observed = 0
            for year in sorted({d.year for d in dates[m]}):
                ym = m & np.array([d.year == year for d in dates])
                ybase = cmask & np.array([d.year == year for d in dates])
                if ym.sum() < 8 or ybase.sum() < 30:
                    continue
                observed += 1
                stable += float(y[ym].mean()) > float(y[ybase].mean())
            rows.append((float(y[m].mean()) / base, month, int(m.sum()), stable, observed))
        top = sorted(rows, reverse=True)[:3]
        rendered = ", ".join(
            f"{dt.date(2000, month, 1):%b} lift {lf:.2f}, устойчивость {stable}/{observed}"
            for lf, month, _n, stable, observed in top
        )
        print(f"  {currency}: {rendered}")


if __name__ == "__main__":
    main()
