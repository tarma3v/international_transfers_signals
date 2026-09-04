"""Shared discrete-time hazard model for the five-publication barrier target.

Training rows are expanded into conditional survival steps.  A row contributes
to step k only if it survived steps 1..k-1.  Step indicators and selected
step-by-state interactions let one pooled model share information across the
five hazards; the final score is the product of the five conditional survival
probabilities.
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

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets, target_now_favourable
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    ALL_YEARS, FINAL_YEARS, GENERAL_YEARS, SHOCK_YEARS,
    _features, _metric_grid, _panel_features, _select,
)

OUT = Path("results/research/round3")
SEED = 20260904


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    window_years: int | None
    local: bool = False


def specs() -> list[Spec]:
    return [
        Spec("pooled_hazard_hist_w3", "hist", 3),
        Spec("pooled_hazard_hist_w5", "hist", 5),
        Spec("pooled_hazard_extra_w5", "extra", 5),
        Spec("pooled_hazard_logit_expand", "logit", None),
        Spec("pooled_hazard_local_hist_w5", "hist", 5, local=True),
    ]


def _model(kind: str):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=240, learning_rate=.035, max_leaf_nodes=11,
            min_samples_leaf=90, l2_regularization=15.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=60,
            max_features=.7, n_jobs=-1, random_state=SEED,
        )
    if kind == "logit":
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(C=.03, max_iter=3000, random_state=SEED),
        )
    raise KeyError(kind)


def _design(X: np.ndarray, rows: np.ndarray, steps: np.ndarray) -> np.ndarray:
    normalized = (steps.astype(float) + 1.0) / 5.0
    onehot = np.eye(5)[steps]
    # Interact the most important state variables (front of compact set) with
    # time-to-barrier while leaving the full compact vector shared.
    interactions = X[rows, :20] * normalized[:, None]
    return np.column_stack([X[rows], onehot, normalized, interactions])


def _expanded_train(X: np.ndarray, alive: np.ndarray, train: np.ndarray):
    rows, steps, targets = [], [], []
    for step in range(5):
        risk = np.ones(len(train), dtype=bool) if step == 0 else alive[step - 1, train] == 1
        selected = train[risk]
        rows.append(selected)
        steps.append(np.full(len(selected), step, dtype=int))
        targets.append(alive[step, selected])
    rows = np.concatenate(rows); steps = np.concatenate(steps); targets = np.concatenate(targets)
    return _design(X, rows, steps), targets


def _predict_survival(model, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    score = np.ones(len(rows), dtype=float)
    for step in range(5):
        design = _design(X, rows, np.full(len(rows), step, dtype=int))
        score *= model.predict_proba(design)[:, 1]
    return score


def generate(spec: Spec, X: np.ndarray, alive: np.ndarray, dates: np.ndarray,
             currencies: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    valid = np.all(np.isfinite(alive), axis=0)
    for year in ALL_YEARS:
        test_start = dt.date(year, 1, 1)
        calib_start = dt.date(year - 1, 1, 1)
        train_mask = np.asarray([r < calib_start for r in reach]) & valid
        if spec.window_years is not None:
            lower = dt.date(year - 1 - spec.window_years, 1, 1)
            train_mask &= dates >= lower
        calib_mask = (dates >= calib_start) & (dates < test_start) & valid
        test_mask = np.asarray([d.year == year for d in dates]) & valid
        tr, ca, te = np.where(train_mask)[0], np.where(calib_mask)[0], np.where(test_mask)[0]
        ca_score = np.full(len(ca), np.nan); te_score = np.full(len(te), np.nan)
        for currency in (CORRIDORS if spec.local else (None,)):
            if currency is None:
                trp, cap, tep = tr, np.arange(len(ca)), np.arange(len(te))
            else:
                trp = tr[currencies[tr] == currency]
                cap = np.where(currencies[ca] == currency)[0]
                tep = np.where(currencies[te] == currency)[0]
            design, target = _expanded_train(X, alive, trp)
            model = _model(spec.kind)
            model.fit(design, target)
            ca_score[cap] = _predict_survival(model, X, ca[cap])
            te_score[tep] = _predict_survival(model, X, te[tep])
        output[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": ca_score, "test_score": te_score}
        print(f"{spec.name:<31} year={year} train={len(tr):5d}", flush=True)
    return output


def _alive(series: dict, index: list[tuple]) -> np.ndarray:
    result = np.full((5, len(index)), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        values = series[currency].values
        for step in range(1, 6):
            value = target_now_favourable(values, i, step)
            if value is not None:
                result[step - 1, row] = value
    return result


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
    X = X[:, _features(names)["compact"]]
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    alive = _alive(series, index)
    benefit = _benefit(series, index)

    outputs = {spec.name: generate(spec, X, alive, dates, currencies, reach)
               for spec in specs()}
    with (OUT / "pooled_hazard_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for name, output in outputs.items():
        rows.extend(_metric_grid(output, y, dates, currencies, benefit, GENERAL_YEARS, name))
    general = pd.DataFrame(rows)
    general.to_csv(OUT / "pooled_hazard_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _n, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "pooled_hazard_stage1.csv", index=False)

    shock_rows = []
    for row in stage1.itertuples(index=False):
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
    shock.to_csv(OUT / "pooled_hazard_stage2_2022_2023.csv", index=False)

    final_rows = []
    for row in shock.head(3).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.stage1_rate), int(row.stage1_rolling) or None,
            int(row.cooldown_days),
        )
        result.update({"candidate": row.candidate,
                       "status": "retrospective; final interval previously inspected"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "pooled_hazard_final_2024_2026_retrospective.csv", index=False)
    (OUT / "pooled_hazard_protocol.json").write_text(json.dumps({
        "specs": [s.__dict__ for s in specs()],
        "target": "conditional survival at each future publication",
        "score": "product of five pooled conditional survival probabilities",
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

