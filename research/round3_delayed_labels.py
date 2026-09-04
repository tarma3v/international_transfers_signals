"""Causal delayed-outcome features for the h=5 forecasting target.

At publication index ``i`` the h=5 label from ``i-5`` has just become fully
observable.  Labels from more recent origins are not used.  The script also
encodes how far the four still-open historical origins have survived using only
prices through ``i``.  These features exploit target persistence without target
leakage.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets, target_now_favourable
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    ALL_YEARS,
    FINAL_YEARS,
    GENERAL_YEARS,
    SHOCK_YEARS,
    _features,
    _metric_grid,
    _panel_features,
    _select,
)

OUT = Path("results/research/round3")
SEED = 20260904


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    local: bool = False
    window_years: int | None = 5
    delayed_only: bool = False


def specs() -> list[Spec]:
    return [
        Spec("delayed_global_hist_w3", "hist", window_years=3),
        Spec("delayed_global_hist_w5", "hist", window_years=5),
        Spec("delayed_global_extra_w5", "extra", window_years=5),
        Spec("delayed_local_hist_w5", "hist", local=True, window_years=5),
        Spec("delayed_global_xgb_w5", "xgb", window_years=5),
        Spec("delayed_global_logit_expand", "logit", window_years=None),
        Spec("delayed_only_hist_w5", "hist", window_years=5, delayed_only=True),
    ]


def _model(kind: str):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=240, learning_rate=.035, max_leaf_nodes=11,
            min_samples_leaf=60, l2_regularization=10.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=35,
            max_features=.70, n_jobs=-1, random_state=SEED,
        )
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=450, max_depth=4, learning_rate=.035,
            min_child_weight=35, subsample=.80, colsample_bytree=.75,
            reg_lambda=8.0, reg_alpha=.3, n_jobs=-1,
            random_state=SEED, eval_metric="logloss",
        )
    if kind == "logit":
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(C=.05, max_iter=3000, random_state=SEED),
        )
    raise KeyError(kind)


def delayed_features(series: dict, index: list[tuple]) -> tuple[np.ndarray, list[str]]:
    names = [f"resolved_fav_lag_{lag}" for lag in range(5, 21)]
    names += [f"resolved_rate_{window}" for window in (5, 10, 20, 60, 120)]
    names += ["resolved_streak", "resolved_transition_rate_20"]
    for lag in range(1, 5):
        names += [f"pending_survival_{lag}", f"pending_margin_{lag}"]
    result = np.zeros((len(index), len(names)), dtype=float)
    for row, (currency, i, _day) in enumerate(index):
        values = series[currency].values
        resolved = []
        # Every origin here reaches no later than the current publication.
        for lag in range(5, 121):
            origin = i - lag
            value = (
                target_now_favourable(values, origin, 5)
                if origin >= 0 else None
            )
            resolved.append(0.5 if value is None else float(value))
        feature = {}
        for lag in range(5, 21):
            feature[f"resolved_fav_lag_{lag}"] = resolved[lag - 5]
        for window in (5, 10, 20, 60, 120):
            feature[f"resolved_rate_{window}"] = float(np.mean(resolved[:window]))
        streak = 1
        while streak < len(resolved) and resolved[streak] == resolved[0]:
            streak += 1
        feature["resolved_streak"] = float(streak if resolved[0] == 1.0 else -streak)
        recent = np.asarray(resolved[:21])
        feature["resolved_transition_rate_20"] = float(np.mean(recent[1:] != recent[:-1]))
        for lag in range(1, 5):
            origin = i - lag
            if origin < 0:
                survival, margin = .5, 0.0
            else:
                observed = values[origin + 1:i + 1]
                survival = float(values[origin] <= np.min(observed))
                margin = (float(np.min(observed)) / float(values[origin]) - 1.0) * 10000.0
            feature[f"pending_survival_{lag}"] = survival
            feature[f"pending_margin_{lag}"] = margin
        result[row] = [feature[name] for name in names]
    return result, names


def _fit_predict(spec: Spec, X: np.ndarray, y: np.ndarray, train: np.ndarray,
                 calib: np.ndarray, test: np.ndarray, currencies: np.ndarray):
    ca_score = np.full(len(calib), np.nan)
    te_score = np.full(len(test), np.nan)
    for currency in (CORRIDORS if spec.local else (None,)):
        if currency is None:
            trp, cap, tep = train, np.arange(len(calib)), np.arange(len(test))
        else:
            trp = train[currencies[train] == currency]
            cap = np.where(currencies[calib] == currency)[0]
            tep = np.where(currencies[test] == currency)[0]
        model = _model(spec.kind)
        model.fit(X[trp], y[trp])
        ca_score[cap] = model.predict_proba(X[calib[cap]])[:, 1]
        te_score[tep] = model.predict_proba(X[test[tep]])[:, 1]
    return ca_score, te_score


def generate(spec: Spec, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
             currencies: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    for year in ALL_YEARS:
        test_start = dt.date(year, 1, 1)
        calib_start = dt.date(year - 1, 1, 1)
        train_mask = np.asarray([r < calib_start for r in reach]) & ~np.isnan(y)
        if spec.window_years is not None:
            lower = dt.date(year - 1 - spec.window_years, 1, 1)
            train_mask &= dates >= lower
        calib_mask = (dates >= calib_start) & (dates < test_start) & ~np.isnan(y)
        test_mask = np.asarray([d.year == year for d in dates]) & ~np.isnan(y)
        tr, ca, te = np.where(train_mask)[0], np.where(calib_mask)[0], np.where(test_mask)[0]
        cs, ts = _fit_predict(spec, X, y, tr, ca, te, currencies)
        output[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": cs, "test_score": ts}
        print(f"{spec.name:<32} year={year} train={len(tr):5d}", flush=True)
    return output


def _benefit(series, index) -> np.ndarray:
    result = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, 5)
        if value is not None:
            result[row] = value
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series = load_or_build()
    X, names = _panel_features(X, names, index)
    delayed, delayed_names = delayed_features(series, index)
    compact_cols = _features(names)["compact"]
    full_matrix = np.column_stack([X[:, compact_cols], delayed])
    delayed_matrix = delayed
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)

    outputs = {}
    for spec in specs():
        matrix = delayed_matrix if spec.delayed_only else full_matrix
        outputs[spec.name] = generate(spec, matrix, y, dates, currencies, reach)
    with (OUT / "delayed_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general_rows = []
    for name, output in outputs.items():
        general_rows.extend(_metric_grid(
            output, y, dates, currencies, benefit, GENERAL_YEARS, name
        ))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "delayed_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "delayed_stage1.csv", index=False)

    shock_rows = []
    for row in stage1.head(5).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, SHOCK_YEARS,
            float(row.rate_target), int(row.rolling_window) or None,
            int(row.cooldown_days),
        )
        result.update({"candidate": row.candidate, "stage1_rate": row.rate_target,
                       "stage1_rolling": row.rolling_window,
                       "stage1_cooldown": row.cooldown_days})
        shock_rows.append(result)
    shock = pd.DataFrame(shock_rows)
    shock["robustness"] = shock[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / "delayed_stage2_2022_2023.csv", index=False)

    final_rows = []
    for row in shock.head(3).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.stage1_rate), int(row.stage1_rolling) or None,
            int(row.stage1_cooldown),
        )
        result.update({"candidate": row.candidate,
                       "status": "retrospective; final interval previously inspected"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "delayed_final_2024_2026_retrospective.csv", index=False)
    (OUT / "delayed_protocol.json").write_text(json.dumps({
        "specs": [s.__dict__ for s in specs()],
        "features": delayed_names,
        "earliest_label_lag": 5,
        "causality": "every resolved label reaches by the row date; pending features use values only through row date",
        "general_years": GENERAL_YEARS,
        "shock_years": SHOCK_YEARS,
        "final_years": FINAL_YEARS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "year_lift_min", "corridor_lift_min", "robustness"]
    print("\nGENERAL", stage1[columns].to_string(index=False), sep="\n")
    print("\nSHOCK", shock[columns].to_string(index=False), sep="\n")
    print("\nFINAL", final[[c for c in columns if c in final]].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()

