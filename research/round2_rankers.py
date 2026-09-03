"""Direct top-tail ranking models for the h=5 alert objective."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    ALL_YEARS, FINAL_YEARS, GENERAL_YEARS, SHOCK_YEARS,
    _features, _metric_grid, _panel_features, _select,
)

OUT = Path("results/research/round2")
SEED = 20260904


@dataclass(frozen=True)
class Spec:
    name: str
    objective: str = "rank:pairwise"
    local: bool = False
    window_years: int = 5
    group_period: str = "quarter"


def specs() -> tuple[Spec, ...]:
    return (
        Spec("global_pairwise_quarter_w3", window_years=3),
        Spec("global_pairwise_quarter_w5", window_years=5),
        Spec("global_pairwise_year_w5", window_years=5, group_period="year"),
        Spec("global_ndcg_quarter_w5", objective="rank:ndcg", window_years=5),
        Spec("local_pairwise_quarter_w5", local=True, window_years=5),
    )


def _model(spec: Spec) -> XGBRanker:
    return XGBRanker(
        objective=spec.objective, n_estimators=320, max_depth=3,
        learning_rate=.03, min_child_weight=35, subsample=.8,
        colsample_bytree=.75, reg_lambda=10.0, reg_alpha=.5,
        n_jobs=-1, random_state=SEED,
    )


def _fit_predict(spec: Spec, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                 currencies: np.ndarray, train: np.ndarray,
                 calib: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if spec.group_period == "quarter":
        period = np.asarray([f"{d.year:04d}Q{(d.month - 1)//3 + 1}" for d in dates])
    else:
        period = np.asarray([str(d.year) for d in dates])
    group_key = np.asarray([f"{c}:{p}" for c, p in zip(currencies, period)])
    order = np.argsort(group_key[train], kind="stable")
    tr = train[order]
    keys, counts = np.unique(group_key[tr], return_counts=True)
    # np.unique sorts keys exactly as argsort above; counts are therefore in the
    # same order as the contiguous training groups.
    assert len(keys) == len(counts) and int(counts.sum()) == len(tr)
    model = _model(spec)
    model.fit(X[tr], y[tr], group=counts.tolist(), verbose=False)
    return model.predict(X[calib]), model.predict(X[test])


def generate(spec: Spec, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
             currencies: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    for year in ALL_YEARS:
        test_start = dt.date(year, 1, 1)
        calibration_start = dt.date(year - 1, 1, 1)
        lower = dt.date(year - 1 - spec.window_years, 1, 1)
        train_mask = np.asarray([r < calibration_start for r in reach]) & ~np.isnan(y)
        train_mask &= dates >= lower
        calib_mask = (dates >= calibration_start) & (dates < test_start) & ~np.isnan(y)
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
            cs, ts = _fit_predict(spec, X, y, dates, currencies,
                                  trp, ca[cap], te[tep])
            ca_score[cap] = cs; te_score[tep] = ts
        output[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": ca_score, "test_score": te_score}
        print(f"{spec.name:<32} year={year} train={len(tr):5d}", flush=True)
    return output


def _benefit(series, index) -> np.ndarray:
    result = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, 5)
        if value is not None: result[row] = value
    return result


def main() -> None:
    X, names, index, series = load_or_build()
    X, names = _panel_features(X, names, index)
    X = X[:, _features(names)["compact"]]
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)
    outputs = {spec.name: generate(spec, X, y, dates, currencies, reach) for spec in specs()}
    with (OUT / "ranker_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    general_rows = []
    for name, output in outputs.items():
        general_rows.extend(_metric_grid(
            output, y, dates, currencies, benefit, GENERAL_YEARS, name,
        ))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "ranker_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "ranker_stage1.csv", index=False)
    shock_rows = []
    for row in stage1.itertuples(index=False):
        result = evaluate(outputs[row.candidate], y, dates, currencies, benefit,
                          SHOCK_YEARS, float(row.rate_target),
                          int(row.rolling_window) or None, int(row.cooldown_days))
        result.update({"candidate": row.candidate, "stage1_rate": row.rate_target,
                       "stage1_rolling": row.rolling_window,
                       "stage1_cooldown": row.cooldown_days})
        shock_rows.append(result)
    shock = pd.DataFrame(shock_rows)
    shock["robustness"] = shock[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / "ranker_stage2_2022_2023.csv", index=False)
    final_rows = []
    for row in shock.head(3).itertuples(index=False):
        result = evaluate(outputs[row.candidate], y, dates, currencies, benefit,
                          FINAL_YEARS, float(row.stage1_rate),
                          int(row.stage1_rolling) or None, int(row.stage1_cooldown))
        result.update({"candidate": row.candidate,
                       "status": "retrospective: final block previously inspected"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "ranker_final_2024_2026_retrospective.csv", index=False)
    (OUT / "ranker_protocol.json").write_text(json.dumps({
        "specs": [s.__dict__ for s in specs()], "general_years": GENERAL_YEARS,
        "shock_years": SHOCK_YEARS, "final_years": FINAL_YEARS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nGENERAL RANKERS\n", stage1[["candidate", "frequency", "lift",
          "forward_benefit_bps", "year_lift_min", "corridor_lift_min"]].to_string(index=False))
    print("\nSHOCK RANKERS\n", shock[["candidate", "frequency", "lift",
          "forward_benefit_bps", "year_lift_min", "corridor_lift_min"]].to_string(index=False))
    print("\nRETROSPECTIVE FINAL\n", final[["candidate", "frequency", "lift",
          "forward_benefit_bps", "year_lift_min", "corridor_lift_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
