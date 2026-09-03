"""Predict the future minimum return instead of the binary local-minimum label.

The binary target fav_h5 is exactly the event `future_min_return >= 0`.  A
regressor preserves information about how far the future minimum is from today,
whereas a classifier sees only 0/1. All reported scores are purged walk-forward.

Run:  .venv/bin/python run_future_drawdown_experiment.py
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.data import CORRIDORS, REFERENCE, load
from ml.features import build_matrix
from ml.targets import build_targets
from ml.validation import assert_no_overlap, target_reach_dates, walk_forward_folds
from run_encoding_experiment import variants as encoding_variants

FIRST_TEST_YEAR = 2021
HORIZON = 5
RANDOM_STATE = 42


def regressors() -> dict[str, object]:
    return {
        "Ridge future-min": Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=20.0)),
        ]),
        "RandomForest future-min": RandomForestRegressor(
            n_estimators=400, max_depth=7, min_samples_leaf=30,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGB future-min": HistGradientBoostingRegressor(
            max_iter=250, max_depth=4, learning_rate=0.04,
            min_samples_leaf=35, l2_regularization=2.0, random_state=RANDOM_STATE,
        ),
        "CatBoost future-min": CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.035, l2_leaf_reg=8,
            loss_function="RMSE", verbose=0, random_seed=RANDOM_STATE,
            allow_writing_files=False,
        ),
        "CatBoost q25 future-min": CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.035, l2_leaf_reg=8,
            loss_function="Quantile:alpha=0.25", verbose=0, random_seed=RANDOM_STATE,
            allow_writing_files=False,
        ),
        "XGBoost future-min": XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.035,
            min_child_weight=25, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=5.0, n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def future_min_target(series, index) -> np.ndarray:
    out = np.full(len(index), np.nan)
    for r, (currency, i, _date) in enumerate(index):
        values = series[currency].values
        if i + HORIZON >= len(values):
            continue
        out[r] = (float(values[i + 1: i + HORIZON + 1].min()) / float(values[i]) - 1.0) * 10000.0
    return out


def main() -> None:
    series = load()
    X_new, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    # The earlier ablation showed that one-hot currency helps while replacing
    # ordinal calendar features with one Fourier harmonic hurts. Use the best
    # representation from that pre-declared ablation for this model-family test.
    X = encoding_variants(X_new, names, index)["только one-hot currency"]
    dates = np.array([d for _c, _i, d in index], dtype=object)
    currencies = np.array([c for c, _i, _d in index], dtype=object)
    y_binary = build_targets(series, index)[f"fav_h{HORIZON}"]
    y_cont = future_min_target(series, index)
    reach = target_reach_dates(index, series, HORIZON)
    folds = walk_forward_folds(dates, FIRST_TEST_YEAR, HORIZON, reach=reach)

    model_scores = {name: np.full(len(index), np.nan) for name in regressors()}
    model_scores["LogReg binary"] = np.full(len(index), np.nan)
    oos = np.zeros(len(index), dtype=bool)
    for train_idx, test_idx, year in folds:
        tr = train_idx[~np.isnan(y_cont[train_idx])]
        te = test_idx[~np.isnan(y_cont[test_idx])]
        assert_no_overlap(dates, tr, te, HORIZON, index=index, series=series)
        transformed = np.arcsinh(y_cont[tr] / 100.0)
        for name, model in regressors().items():
            model.fit(X[tr], transformed)
            model_scores[name][te] = model.predict(X[te])
        classifier = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)),
        ]).fit(X[tr], y_binary[tr])
        model_scores["LogReg binary"][te] = classifier.predict_proba(X[te])[:, 1]
        oos[te] = True
        print(f"  fold {year}: train={len(tr)}, test={len(te)}", flush=True)

    model_scores["pct_range_90"] = X_new[:, names.index("pct_range_90")]
    valid_base = oos & ~np.isnan(y_binary)
    print("\nAUC по событию future_min_return >= 0, h=5")
    print(f"{'модель':<29}{'ВСЕ':>8}" + "".join(f"{c:>8}" for c in CORRIDORS) + "  лет >0.5")
    for name, score in model_scores.items():
        valid = valid_base & ~np.isnan(score)
        pooled = roc_auc_score(y_binary[valid], score[valid])
        per_currency = []
        for currency in CORRIDORS:
            m = valid & (currencies == currency)
            per_currency.append(roc_auc_score(y_binary[m], score[m]))
        better = total = 0
        for year in sorted({d.year for d in dates[valid]}):
            m = valid & np.array([d.year == year for d in dates])
            if len(np.unique(y_binary[m])) < 2:
                continue
            total += 1
            better += roc_auc_score(y_binary[m], score[m]) > 0.5
        print(f"{name:<29}{pooled:>8.3f}" + "".join(f"{v:>8.3f}" for v in per_currency)
              + f"  {better}/{total}")


if __name__ == "__main__":
    main()
