"""Diverse, non-anchor model families for deep-research round two.

The script tests local, global, partially pooled, analogue, regime-mixture,
discrete-survival and future-floor approaches.  Every test year has a separate
preceding calibration year, and model training is purged by the actual h=5
target-reach date.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets, target_now_favourable
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate

OUT = Path("results/research/round2")
SEED = 20260904
GENERAL_YEARS = (2017, 2018, 2019, 2020)
SHOCK_YEARS = (2022, 2023)
FINAL_YEARS = (2024, 2025, 2026)
ALL_YEARS = GENERAL_YEARS + SHOCK_YEARS + FINAL_YEARS


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[], object] | None
    feature_set: str
    local: bool = False
    window_years: int | None = None
    mode: str = "binary"


def _panel_features(X, names, index):
    """Add same-date and lagged common factors without reading future rows."""
    base = pd.DataFrame({
        "date": [d for _c, _i, d in index],
        "currency": [c for c, _i, _d in index],
    })
    source = ("ret_1", "ret_5", "ret_20", "ret_60", "raw_vol_20")
    for name in source:
        base[name] = X[:, names.index(name)]
    dates = sorted(base.date.unique())
    date_pos = {day: i for i, day in enumerate(dates)}
    additions = {}
    for name in source:
        common = base.groupby("date")[name].mean().reindex(dates)
        additions[f"common_{name}"] = base.date.map(common).to_numpy(float)
        if name.startswith("ret_"):
            additions[f"residual_{name}"] = base[name].to_numpy(float) - additions[f"common_{name}"]
    common_ret1 = pd.Series(
        base.groupby("date").ret_1.mean().reindex(dates).to_numpy(), index=dates
    )
    additions["common_ret1_vol20"] = base.date.map(common_ret1.rolling(20, min_periods=5).std()).fillna(0).to_numpy(float)
    additions["common_ret1_mean5"] = base.date.map(common_ret1.rolling(5, min_periods=1).mean()).to_numpy(float)
    additions["common_ret1_mean20"] = base.date.map(common_ret1.rolling(20, min_periods=1).mean()).to_numpy(float)
    for lag in (1, 2, 3, 5):
        lagged = common_ret1.shift(lag)
        additions[f"common_ret1_lag{lag}"] = base.date.map(lagged).fillna(0).to_numpy(float)
    extra_names = list(additions)
    extra = np.column_stack([additions[name] for name in extra_names])
    return np.column_stack([X, extra]), list(names) + extra_names


def _features(names):
    currency = [n for n in names if n.startswith("currency_")]
    compact = [
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
        "range_pos_20", "range_pos_60", "range_pos_120", "range_pos_250",
        "slope_z_20", "slope_z_60", "raw_vol_5", "raw_vol_20", "raw_vol_60",
        "vol_ratio_20_120", "positive_share_20", "ret_ac1_20",
        "streak_up", "streak_dn", "bars_since_min_30", "bars_since_max_30",
        "common_ret_1", "common_ret_5", "common_ret_20", "common_ret_60",
        "residual_ret_1", "residual_ret_5", "residual_ret_20", "residual_ret_60",
        "common_ret1_vol20", "common_ret1_mean5", "common_ret1_mean20",
        "common_ret1_lag1", "common_ret1_lag2", "common_ret1_lag3", "common_ret1_lag5",
        "peer_dispersion_5", "rel_to_peers_5", "usd_ret_5", "usd_ret_20",
        "cny_ret_5", "cny_ret_20", "annual_sin_1", "annual_cos_1",
        "annual_sin_2", "annual_cos_2", "dow_sin", "dow_cos", "gap_days",
    ]
    spline = [
        "ret_1", "ret_5", "ret_20", "ret_60", "range_pos_20", "range_pos_60",
        "slope_z_20", "raw_vol_20", "vol_ratio_20_120", "common_ret_5",
        "common_ret_20", "residual_ret_5", "residual_ret_20",
    ]
    regime = [
        "ret_20", "ret_60", "slope_z_20", "slope_z_60", "raw_vol_20",
        "raw_vol_60", "vol_ratio_20_120", "common_ret_20", "common_ret_60",
        "common_ret1_vol20", "residual_ret_20", "peer_dispersion_5",
    ]
    return {
        "compact": np.asarray([names.index(n) for n in compact + currency]),
        "spline": np.asarray([names.index(n) for n in spline]),
        "regime": np.asarray([names.index(n) for n in regime + currency]),
    }


def _logit():
    return make_pipeline(
        RobustScaler(),
        LogisticRegression(C=0.08, max_iter=2500, class_weight=None, random_state=SEED),
    )


def _spline_logit():
    return make_pipeline(
        RobustScaler(),
        SplineTransformer(n_knots=4, degree=2, include_bias=False),
        LogisticRegression(C=0.04, max_iter=3000, random_state=SEED),
    )


def _knn(k=150):
    return make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance", p=2)
    )


def _lda():
    return make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )


def _hist():
    return HistGradientBoostingClassifier(
        max_iter=220, learning_rate=0.035, max_leaf_nodes=9,
        min_samples_leaf=70, l2_regularization=12.0, random_state=SEED,
    )


def _extra():
    return ExtraTreesClassifier(
        n_estimators=500, max_depth=7, min_samples_leaf=40,
        max_features=0.7, n_jobs=-1, random_state=SEED,
    )


def _quantile(alpha=.35):
    return GradientBoostingRegressor(
        loss="quantile", alpha=alpha, n_estimators=180, max_depth=2,
        min_samples_leaf=70, learning_rate=0.035, random_state=SEED,
    )


def candidates():
    return [
        Candidate("local_compact_logit", _logit, "compact", local=True),
        Candidate("global_compact_logit", _logit, "compact"),
        Candidate("local_spline_logit", _spline_logit, "spline", local=True),
        Candidate("global_spline_logit", _spline_logit, "spline"),
        Candidate("local_path_knn150", lambda: _knn(150), "compact", local=True, window_years=7),
        Candidate("local_path_knn300", lambda: _knn(300), "compact", local=True, window_years=7),
        Candidate("global_path_knn300", lambda: _knn(300), "compact", window_years=7),
        Candidate("local_shrinkage_lda", _lda, "compact", local=True),
        Candidate("global_shrinkage_lda", _lda, "compact"),
        Candidate("local_compact_hist", _hist, "compact", local=True, window_years=7),
        Candidate("global_compact_hist", _hist, "compact", window_years=7),
        Candidate("local_compact_extra", _extra, "compact", local=True, window_years=7),
        Candidate("global_compact_extra", _extra, "compact", window_years=7),
        Candidate("global_gmm3_hist", None, "regime", window_years=7, mode="gmm"),
        Candidate("global_survival_logit", _logit, "compact", mode="survival"),
        Candidate("local_survival_logit", _logit, "compact", local=True, mode="survival"),
        Candidate("global_floor_q25", lambda: _quantile(.25), "compact", window_years=7, mode="floor"),
        Candidate("global_floor_q50", lambda: _quantile(.50), "compact", window_years=7, mode="floor"),
        Candidate("local_floor_q35", lambda: _quantile(.35), "compact", local=True, window_years=7, mode="floor"),
    ]


def _fit_predict_plain(candidate, X, y, train, calib, test):
    model = candidate.factory()
    model.fit(X[train], y[train])
    return model.predict_proba(X[calib])[:, 1], model.predict_proba(X[test])[:, 1]


def _fit_predict_gmm(X, y, train, calib, test):
    scaler = StandardScaler().fit(X[train])
    tr = scaler.transform(X[train]); ca = scaler.transform(X[calib]); te = scaler.transform(X[test])
    gmm = GaussianMixture(n_components=3, covariance_type="diag", reg_covar=1e-4,
                          random_state=SEED, n_init=5).fit(tr)
    tr_aug = np.column_stack([tr, gmm.predict_proba(tr)])
    ca_aug = np.column_stack([ca, gmm.predict_proba(ca)])
    te_aug = np.column_stack([te, gmm.predict_proba(te)])
    model = _hist(); model.fit(tr_aug, y[train])
    return model.predict_proba(ca_aug)[:, 1], model.predict_proba(te_aug)[:, 1]


def _fit_predict_survival(candidate, X, alive, train, calib, test):
    ca_score = np.ones(len(calib)); te_score = np.ones(len(test))
    for step in range(1, 6):
        at_risk = train if step == 1 else train[alive[step - 2][train] == 1]
        target = alive[step - 1]
        if candidate.local:
            raise RuntimeError("local survival is dispatched by currency outside")
        model = candidate.factory(); model.fit(X[at_risk], target[at_risk])
        ca_score *= model.predict_proba(X[calib])[:, 1]
        te_score *= model.predict_proba(X[test])[:, 1]
    return ca_score, te_score


def _fit_predict_floor(candidate, X, floor, train, calib, test):
    model = candidate.factory(); model.fit(X[train], floor[train])
    return model.predict(X[calib]), model.predict(X[test])


def generate(candidate, X, cols, y, alive, floor, dates, currencies, reach):
    matrix = X[:, cols]
    output = {}
    for year in ALL_YEARS:
        test_start = dt.date(year, 1, 1)
        calib_start = dt.date(year - 1, 1, 1)
        train = np.asarray([r < calib_start for r in reach]) & ~np.isnan(y)
        if candidate.window_years:
            lower = dt.date(year - 1 - candidate.window_years, 1, 1)
            train &= np.asarray([d >= lower for d in dates])
        calib = np.asarray([calib_start <= d < test_start for d in dates]) & ~np.isnan(y)
        test = np.asarray([d.year == year for d in dates]) & ~np.isnan(y)
        tr, ca, te = np.where(train)[0], np.where(calib)[0], np.where(test)[0]
        if min(len(tr), len(ca), len(te)) == 0:
            continue
        ca_score = np.full(len(ca), np.nan); te_score = np.full(len(te), np.nan)
        groups = CORRIDORS if candidate.local else (None,)
        for currency in groups:
            if currency is None:
                trp, cap, tep = tr, np.arange(len(ca)), np.arange(len(te))
            else:
                trp = tr[currencies[tr] == currency]
                cap = np.where(currencies[ca] == currency)[0]
                tep = np.where(currencies[te] == currency)[0]
            if min(len(trp), len(cap), len(tep)) == 0:
                continue
            if candidate.mode == "binary":
                cs, ts = _fit_predict_plain(candidate, matrix, y, trp, ca[cap], te[tep])
            elif candidate.mode == "gmm":
                cs, ts = _fit_predict_gmm(matrix, y, trp, ca[cap], te[tep])
            elif candidate.mode == "floor":
                cs, ts = _fit_predict_floor(candidate, matrix, floor, trp, ca[cap], te[tep])
            elif candidate.mode == "survival":
                local_candidate = Candidate(candidate.name, candidate.factory, candidate.feature_set, False)
                cs, ts = _fit_predict_survival(local_candidate, matrix, alive, trp, ca[cap], te[tep])
            else:
                raise KeyError(candidate.mode)
            ca_score[cap], te_score[tep] = cs, ts
        if np.all(np.isfinite(ca_score)) and np.all(np.isfinite(te_score)):
            output[year] = {"calib_idx": ca, "test_idx": te,
                            "calib_score": ca_score, "test_score": te_score}
        print(f"{candidate.name:<28} year={year} train={len(tr):5d}", flush=True)
    return output


def _future_objects(series, index):
    benefit = np.full(len(index), np.nan); floor = np.full(len(index), np.nan)
    alive = [np.full(len(index), np.nan) for _ in range(5)]
    for row, (currency, i, _day) in enumerate(index):
        values = series[currency].values
        b = benefit_forward_only(values, i, 5)
        if b is not None:
            benefit[row] = b
            floor[row] = (np.min(values[i + 1:i + 6]) / values[i] - 1.0) * 10000.0
        for step in range(1, 6):
            v = target_now_favourable(values, i, step)
            if v is not None:
                alive[step - 1][row] = v
    return benefit, floor, alive


def _metric_grid(outputs, y, dates, currencies, benefit, years, candidate):
    rows = []
    for rate in (0.20, 0.25, 0.30, 0.35, 0.40):
        for rolling, cooldown in ((None, 0), (120, 0), (250, 0), (500, 0), (250, 3)):
            row = evaluate(outputs, y, dates, currencies, benefit, years, rate, rolling, cooldown)
            row["candidate"] = candidate; rows.append(row)
    return rows


def _select(frame):
    z = frame[
        frame.frequency.between(.90, 2.10)
        & (frame.corridor_freq_min >= .65)
        & (frame.forward_benefit_bps > 0)
    ].copy()
    if not len(z): z = frame.copy()
    z["robustness"] = z[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    return z.sort_values(["robustness", "lift", "auc"], ascending=False).iloc[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series = load_or_build()
    X, names = _panel_features(X, names, index)
    feature_sets = _features(names)
    dates = np.asarray([d for _c, _i, d in index], dtype=object)
    currencies = np.asarray([c for c, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit, floor, alive = _future_objects(series, index)

    prediction_path = OUT / "diverse_outputs.pkl"
    if prediction_path.exists():
        with prediction_path.open("rb") as fh: outputs = pickle.load(fh)
    else:
        outputs = {}
        for candidate in candidates():
            outputs[candidate.name] = generate(
                candidate, X, feature_sets[candidate.feature_set], y, alive, floor,
                dates, currencies, reach,
            )
        with prediction_path.open("wb") as fh:
            pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general_rows = []
    for candidate in candidates():
        general_rows.extend(_metric_grid(
            outputs[candidate.name], y, dates, currencies, benefit,
            GENERAL_YEARS, candidate.name,
        ))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "diverse_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "diverse_stage1.csv", index=False)

    # Only the eight strongest architectures from general validation proceed.
    shock_rows = []
    for row in stage1.head(8).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, SHOCK_YEARS,
            float(row.rate_target), int(row.rolling_window) or None, int(row.cooldown_days),
        )
        result.update({"candidate": row.candidate,
                       "stage1_rate": row.rate_target,
                       "stage1_rolling": row.rolling_window,
                       "stage1_cooldown": row.cooldown_days})
        shock_rows.append(result)
    shock = pd.DataFrame(shock_rows)
    shock["robustness"] = shock[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / "diverse_stage2_2022_2023.csv", index=False)

    final_rows = []
    for row in shock.head(4).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.stage1_rate), int(row.stage1_rolling) or None, int(row.stage1_cooldown),
        )
        result.update({"candidate": row.candidate,
                       "status": "retrospective: final block previously inspected"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "diverse_final_2024_2026_retrospective.csv", index=False)
    (OUT / "diverse_protocol.json").write_text(json.dumps({
        "general_years": GENERAL_YEARS, "shock_years": SHOCK_YEARS,
        "final_years": FINAL_YEARS, "candidates": [c.__dict__ | {"factory": None} for c in candidates()],
        "feature_sets": {k: [names[i] for i in v] for k, v in feature_sets.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nGENERAL ARCHITECTURES")
    print(stage1[["candidate", "frequency", "lift", "forward_benefit_bps",
                  "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nSHOCK VALIDATION")
    print(shock[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nRETROSPECTIVE FINAL")
    print(final[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min"]].to_string(index=False))


if __name__ == "__main__":
    main()

