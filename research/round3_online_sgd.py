"""Causal online logistic models with five-publication delayed labels.

Unlike the yearly batch models, these classifiers update their coefficients
inside the calibration and forecast years.  A prediction can train the model
only once its fifth future publication has arrived.  This directly targets the
observed structural instability while preserving the operational information
set.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    ALL_YEARS, FINAL_YEARS, GENERAL_YEARS, SHOCK_YEARS,
    _features, _metric_grid, _panel_features, _select,
)
from research.round3_delayed_labels import delayed_features

OUT = Path("results/research/round3")
SEED = 20260904


@dataclass(frozen=True)
class Spec:
    name: str
    eta0: float
    update_epochs: int
    window_years: int | None
    local: bool = False
    delayed: bool = True


def specs() -> list[Spec]:
    return [
        Spec("online_sgd_w5_eta005_e1", .005, 1, 5),
        Spec("online_sgd_w5_eta02_e1", .02, 1, 5),
        Spec("online_sgd_w5_eta05_e1", .05, 1, 5),
        Spec("online_sgd_w5_eta005_e3", .005, 3, 5),
        Spec("online_sgd_w5_eta02_e3", .02, 3, 5),
        Spec("online_sgd_expand_eta02_e1", .02, 1, None),
        Spec("online_sgd_local_w5_eta02_e1", .02, 1, 5, local=True),
        Spec("online_sgd_compact_w5_eta02_e1", .02, 1, 5, delayed=False),
    ]


def _new_model(spec: Spec) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=2e-4,
        learning_rate="constant", eta0=spec.eta0,
        max_iter=1200, tol=1e-4, random_state=SEED,
    )


def _stream(spec: Spec, X: np.ndarray, y: np.ndarray, reach: np.ndarray,
            dates: np.ndarray, train: np.ndarray, calib: np.ndarray,
            test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X[train])
    model = _new_model(spec)
    model.fit(scaler.transform(X[train]), y[train])

    rows = np.concatenate([calib, test])
    roles = np.concatenate([
        np.zeros(len(calib), dtype=int), np.ones(len(test), dtype=int)
    ])
    original_roles = roles.copy()
    order = np.argsort(dates[rows], kind="stable")
    rows, roles = rows[order], roles[order]
    score = np.full(len(rows), np.nan)
    pending: list[int] = []
    start = 0
    while start < len(rows):
        day = dates[rows[start]]
        stop = start + 1
        while stop < len(rows) and dates[rows[stop]] == day:
            stop += 1
        ready = [pos for pos in pending if reach[rows[pos]] <= day]
        if ready:
            update_x = scaler.transform(X[rows[ready]])
            update_y = y[rows[ready]]
            for _ in range(spec.update_epochs):
                model.partial_fit(update_x, update_y, classes=np.asarray([0.0, 1.0]))
            ready_set = set(ready)
            pending = [pos for pos in pending if pos not in ready_set]
        score[start:stop] = model.predict_proba(
            scaler.transform(X[rows[start:stop]])
        )[:, 1]
        pending.extend(range(start, stop))
        start = stop

    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    restored = score[inverse]
    return restored[original_roles == 0], restored[original_roles == 1]


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
        ca_score = np.full(len(ca), np.nan); te_score = np.full(len(te), np.nan)
        for currency in (CORRIDORS if spec.local else (None,)):
            if currency is None:
                trp, cap, tep = tr, np.arange(len(ca)), np.arange(len(te))
            else:
                trp = tr[currencies[tr] == currency]
                cap = np.where(currencies[ca] == currency)[0]
                tep = np.where(currencies[te] == currency)[0]
            cs, ts = _stream(spec, X, y, reach, dates, trp, ca[cap], te[tep])
            ca_score[cap], te_score[tep] = cs, ts
        output[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": ca_score, "test_score": te_score}
        print(f"{spec.name:<34} year={year} train={len(tr):5d}", flush=True)
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
    compact = X[:, _features(names)["compact"]]
    delayed, delayed_names = delayed_features(series, index)
    full = np.column_stack([compact, delayed])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)

    outputs = {}
    for spec in specs():
        outputs[spec.name] = generate(
            spec, full if spec.delayed else compact, y, dates, currencies, reach
        )
    with (OUT / "online_sgd_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for name, output in outputs.items():
        rows.extend(_metric_grid(output, y, dates, currencies, benefit, GENERAL_YEARS, name))
    general = pd.DataFrame(rows)
    general.to_csv(OUT / "online_sgd_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _n, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "online_sgd_stage1.csv", index=False)

    shock_rows = []
    for row in stage1.head(6).itertuples(index=False):
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
    shock.to_csv(OUT / "online_sgd_stage2_2022_2023.csv", index=False)

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
    final.to_csv(OUT / "online_sgd_final_2024_2026_retrospective.csv", index=False)
    (OUT / "online_sgd_protocol.json").write_text(json.dumps({
        "specs": [s.__dict__ for s in specs()], "delayed_features": delayed_names,
        "feedback_rule": "partial_fit only when target_reach_date <= current publication date",
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

