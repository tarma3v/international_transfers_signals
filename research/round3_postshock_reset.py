"""Post-2022 reset sensitivity for the current deployment regime.

This experiment deliberately discards pre-2022 labels.  It cannot be validated
on the onset of the 2022 break, so every result is retrospective.  Within the
post-break era, 2024 selects the alert policy, 2025 is the model gate, and 2026
is reported last.  The split is chronological but not pristine because earlier
research rounds have already inspected all three years.
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
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import _features, _metric_grid, _panel_features, _select
from research.round3_delayed_labels import delayed_features

OUT = Path("results/research/round3")
SEED = 20260904
RESET = dt.date(2022, 2, 24)
YEARS = (2024, 2025, 2026)


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    delayed: bool = True
    local: bool = False


def specs() -> list[Spec]:
    return [
        Spec("reset_hist", "hist"),
        Spec("reset_hist_compact", "hist", delayed=False),
        Spec("reset_extra", "extra"),
        Spec("reset_xgb", "xgb"),
        Spec("reset_logit", "logit"),
        Spec("reset_local_hist", "hist", local=True),
    ]


def _model(kind: str):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.035, max_leaf_nodes=9,
            min_samples_leaf=40, l2_regularization=15.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=25,
            max_features=.7, n_jobs=-1, random_state=SEED,
        )
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=.03,
            min_child_weight=25, subsample=.8, colsample_bytree=.75,
            reg_lambda=12.0, reg_alpha=.5, n_jobs=-1,
            random_state=SEED, eval_metric="logloss",
        )
    if kind == "logit":
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(C=.03, max_iter=3000, random_state=SEED),
        )
    raise KeyError(kind)


def generate(spec: Spec, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
             currencies: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    for year in YEARS:
        test_start = dt.date(year, 1, 1)
        calib_start = dt.date(year - 1, 1, 1)
        train_mask = (
            (dates >= RESET)
            & np.asarray([r < calib_start for r in reach])
            & ~np.isnan(y)
        )
        calib_mask = (dates >= calib_start) & (dates < test_start) & ~np.isnan(y)
        test_mask = np.asarray([d.year == year for d in dates]) & ~np.isnan(y)
        tr, ca, te = np.where(train_mask)[0], np.where(calib_mask)[0], np.where(test_mask)[0]
        ca_score = np.full(len(ca), np.nan)
        te_score = np.full(len(te), np.nan)
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
    compact = X[:, _features(names)["compact"]]
    delayed, delayed_names = delayed_features(series, index)
    full = np.column_stack([compact, delayed])
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)

    outputs = {}
    for spec in specs():
        outputs[spec.name] = generate(
            spec, full if spec.delayed else compact, y, dates, currencies, reach
        )
    with (OUT / "postshock_reset_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    stage1_rows = []
    for name, output in outputs.items():
        stage1_rows.extend(_metric_grid(output, y, dates, currencies, benefit, (2024,), name))
    stage1_grid = pd.DataFrame(stage1_rows)
    stage1_grid.to_csv(OUT / "postshock_reset_2024_grid.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in stage1_grid.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "postshock_reset_stage1_2024.csv", index=False)

    stage2_rows = []
    for row in stage1.itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, (2025,),
            float(row.rate_target), int(row.rolling_window) or None,
            int(row.cooldown_days),
        )
        result.update({"candidate": row.candidate, "stage1_rate": row.rate_target,
                       "stage1_rolling": row.rolling_window,
                       "stage1_cooldown": row.cooldown_days})
        stage2_rows.append(result)
    stage2 = pd.DataFrame(stage2_rows)
    stage2["robustness"] = stage2[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage2 = stage2.sort_values(["robustness", "lift"], ascending=False)
    stage2.to_csv(OUT / "postshock_reset_stage2_2025.csv", index=False)

    final_rows = []
    for row in stage2.head(3).itertuples(index=False):
        result = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, (2026,),
            float(row.stage1_rate), int(row.stage1_rolling) or None,
            int(row.stage1_cooldown),
        )
        result.update({"candidate": row.candidate,
                       "status": "chronological retrospective; 2026 was inspected in earlier rounds"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "postshock_reset_2026_retrospective.csv", index=False)
    (OUT / "postshock_reset_protocol.json").write_text(json.dumps({
        "reset": str(RESET), "specs": [s.__dict__ for s in specs()],
        "selection_year": 2024, "gate_year": 2025, "reported_last_year": 2026,
        "delayed_features": delayed_names,
        "limitation": "all post-2022 results are retrospective; no untouched post-break holdout remains",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "year_lift_min", "corridor_lift_min", "robustness"]
    print("\n2024 POLICY SELECTION", stage1[columns].to_string(index=False), sep="\n")
    print("\n2025 MODEL GATE", stage2[columns].to_string(index=False), sep="\n")
    print("\n2026 LAST READ", final[[c for c in columns if c in final]].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()

