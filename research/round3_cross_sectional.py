"""Same-publication cross-sectional state features.

CBR corridor rates are observed together.  For each date this experiment adds
panel breadth, dispersion, medians, and the corridor's relative rank for causal
price-position, momentum, and volatility variables.  No later publication or
future label enters the transform.
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
from xgboost import XGBClassifier

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    ALL_YEARS, FINAL_YEARS, GENERAL_YEARS, SHOCK_YEARS,
    _features, _metric_grid, _panel_features, _select,
)

OUT = Path("results/research/round3")
SEED = 20260904
SOURCE = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "days_beaten_30", "days_beaten_90", "days_beaten_180",
    "range_pos_20", "range_pos_60", "range_pos_120", "range_pos_250",
    "ret_1", "ret_5", "ret_20", "ret_60",
    "slope_z_20", "slope_z_60", "raw_vol_20", "vol_ratio_20_120",
    "bars_since_min_30", "bars_since_max_30",
)


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    window_years: int
    local: bool = False


def specs() -> list[Spec]:
    return [
        Spec("cross_hist_w3", "hist", 3),
        Spec("cross_hist_w5", "hist", 5),
        Spec("cross_extra_w3", "extra", 3),
        Spec("cross_extra_w5", "extra", 5),
        Spec("cross_xgb_w5", "xgb", 5),
        Spec("cross_local_hist_w5", "hist", 5, local=True),
    ]


def _model(kind: str):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=240, learning_rate=.035, max_leaf_nodes=11,
            min_samples_leaf=60, l2_regularization=12.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=35,
            max_features=.65, n_jobs=-1, random_state=SEED,
        )
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=450, max_depth=4, learning_rate=.035,
            min_child_weight=35, subsample=.8, colsample_bytree=.7,
            reg_lambda=10.0, reg_alpha=.3, n_jobs=-1,
            random_state=SEED, eval_metric="logloss",
        )
    raise KeyError(kind)


def cross_features(X: np.ndarray, names: list[str], index: list[tuple]):
    frame = pd.DataFrame({"date": [row[2] for row in index]})
    additions = {}
    for name in SOURCE:
        values = pd.Series(X[:, names.index(name)])
        grouped = values.groupby(frame.date)
        mean = grouped.transform("mean")
        additions[f"panel_mean_{name}"] = mean.to_numpy(float)
        additions[f"panel_std_{name}"] = grouped.transform("std").fillna(0).to_numpy(float)
        additions[f"panel_min_{name}"] = grouped.transform("min").to_numpy(float)
        additions[f"panel_max_{name}"] = grouped.transform("max").to_numpy(float)
        additions[f"relative_{name}"] = values.to_numpy(float) - mean.to_numpy(float)
        additions[f"panel_rank_{name}"] = grouped.rank(pct=True).to_numpy(float)
    for name in ("ret_1", "ret_5", "ret_20", "ret_60"):
        positive = pd.Series((X[:, names.index(name)] > 0).astype(float))
        additions[f"panel_positive_share_{name}"] = (
            positive.groupby(frame.date).transform("mean").to_numpy(float)
        )
    extra_names = list(additions)
    extra = np.column_stack([additions[name] for name in extra_names])
    return extra, extra_names


def generate(spec: Spec, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
             currencies: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    for year in ALL_YEARS:
        test_start = dt.date(year, 1, 1)
        calib_start = dt.date(year - 1, 1, 1)
        lower = dt.date(year - 1 - spec.window_years, 1, 1)
        train_mask = (
            np.asarray([r < calib_start for r in reach])
            & (dates >= lower) & ~np.isnan(y)
        )
        calib_mask = (dates >= calib_start) & (dates < test_start) & ~np.isnan(y)
        test_mask = np.asarray([d.year == year for d in dates]) & ~np.isnan(y)
        tr, ca, te = np.where(train_mask)[0], np.where(calib_mask)[0], np.where(test_mask)[0]
        ca_score = np.full(len(ca), np.nan); te_score = np.full(len(te), np.nan)
        for currency in (CORRIDORS if spec.local else (None,)):
            if currency is None:
                trp, cap, tep = tr, np.arange(len(ca)), np.arange(len(te))
            else:
                trp = tr[currencies[tr] == currency]
                cap = np.where(currencies[ca] == currency)[0]
                tep = np.where(currencies[te] == currency)[0]
            model = _model(spec.kind)
            model.fit(X[trp], y[trp])
            ca_score[cap] = model.predict_proba(X[ca[cap]])[:, 1]
            te_score[tep] = model.predict_proba(X[te[tep]])[:, 1]
        output[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": ca_score, "test_score": te_score}
        print(f"{spec.name:<24} year={year} train={len(tr):5d}", flush=True)
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
    cross, cross_names = cross_features(X, names, index)
    compact = X[:, _features(names)["compact"]]
    matrix = np.column_stack([compact, cross])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)

    outputs = {spec.name: generate(spec, matrix, y, dates, currencies, reach)
               for spec in specs()}
    with (OUT / "cross_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for name, output in outputs.items():
        rows.extend(_metric_grid(output, y, dates, currencies, benefit, GENERAL_YEARS, name))
    general = pd.DataFrame(rows)
    general.to_csv(OUT / "cross_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _n, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "cross_stage1.csv", index=False)

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
    shock.to_csv(OUT / "cross_stage2_2022_2023.csv", index=False)

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
    final.to_csv(OUT / "cross_final_2024_2026_retrospective.csv", index=False)
    (OUT / "cross_protocol.json").write_text(json.dumps({
        "specs": [s.__dict__ for s in specs()], "source_features": SOURCE,
        "generated_features": cross_names,
        "information_set": "same-date published CBR corridor panel only",
        "general_years": GENERAL_YEARS, "shock_years": SHOCK_YEARS,
        "final_years": FINAL_YEARS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "year_lift_min", "corridor_lift_min", "robustness"]
    print("\nGENERAL", stage1[columns].to_string(index=False), sep="\n")
    print("\nSHOCK", shock[columns].to_string(index=False), sep="\n")
    print("\nFINAL", final[[c for c in columns if c in final]].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()

