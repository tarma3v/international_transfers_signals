"""Causal recency and post-2022 weighting experiments.

The experiment asks whether model failure after a structural break is better
handled by shorter estimation windows or explicit weighting of recent rows.
All training labels are purged by their actual h=5 target-reach date.  The
post-2022 boost is a user-proposed, predeclared sensitivity and is reported as
retrospective because no untouched post-2022 block remains.
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

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.model_study import combine_outputs, evaluate
from research.round2_diverse_models import (
    ALL_YEARS, FINAL_YEARS, GENERAL_YEARS, SHOCK_YEARS,
    _features, _metric_grid, _panel_features, _select,
)

OUT = Path("results/research/round2")
SEED = 20260904
SHOCK_DATE = dt.date(2022, 2, 24)


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    local: bool = False
    window_years: int | None = None
    half_life_years: float | None = None
    postshock_boost: float = 1.0


def _model(kind: str):
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=350, max_depth=7, min_samples_leaf=40,
            max_features=.70, n_jobs=-1, random_state=SEED,
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=220, learning_rate=.035, max_leaf_nodes=9,
            min_samples_leaf=70, l2_regularization=12.0, random_state=SEED,
        )
    raise KeyError(kind)


def specs() -> list[Spec]:
    result = []
    for kind in ("extra", "hist"):
        for window in (2, 3, 5):
            result.append(Spec(f"global_{kind}_window{window}", kind, window_years=window))
        for half_life in (1.0, 2.0):
            tag = str(half_life).replace(".", "p")
            result.append(Spec(f"global_{kind}_decay{tag}", kind,
                               half_life_years=half_life))
        result.append(Spec(f"local_{kind}_window3", kind, local=True, window_years=3))
        result.append(Spec(f"global_{kind}_postshock4_decay2", kind,
                           half_life_years=2.0, postshock_boost=4.0))
    return result


def _weights(spec: Spec, train: np.ndarray, dates: np.ndarray,
             calibration_start: dt.date) -> np.ndarray | None:
    if spec.half_life_years is None and spec.postshock_boost == 1.0:
        return None
    result = np.ones(len(train), dtype=float)
    if spec.half_life_years is not None:
        age = np.asarray([(calibration_start - dates[row]).days for row in train])
        result *= np.power(.5, age / (365.25 * spec.half_life_years))
    if spec.postshock_boost != 1.0:
        result *= np.where(dates[train] >= SHOCK_DATE, spec.postshock_boost, 1.0)
    # Preserve a stable effective scale across years and estimators.
    return result / max(1e-12, float(np.mean(result)))


def generate(spec: Spec, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
             currencies: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    for year in ALL_YEARS:
        test_start = dt.date(year, 1, 1)
        calibration_start = dt.date(year - 1, 1, 1)
        train_mask = np.asarray([r < calibration_start for r in reach]) & ~np.isnan(y)
        if spec.window_years is not None:
            lower = dt.date(year - 1 - spec.window_years, 1, 1)
            train_mask &= dates >= lower
        calib_mask = (dates >= calibration_start) & (dates < test_start) & ~np.isnan(y)
        test_mask = np.asarray([d.year == year for d in dates]) & ~np.isnan(y)
        tr, ca, te = np.where(train_mask)[0], np.where(calib_mask)[0], np.where(test_mask)[0]
        if min(len(tr), len(ca), len(te)) == 0:
            continue
        ca_score = np.full(len(ca), np.nan); te_score = np.full(len(te), np.nan)
        for currency in (CORRIDORS if spec.local else (None,)):
            if currency is None:
                trp, cap, tep = tr, np.arange(len(ca)), np.arange(len(te))
            else:
                trp = tr[currencies[tr] == currency]
                cap = np.where(currencies[ca] == currency)[0]
                tep = np.where(currencies[te] == currency)[0]
            model = _model(spec.kind)
            model.fit(X[trp], y[trp], sample_weight=_weights(spec, trp, dates, calibration_start))
            ca_score[cap] = model.predict_proba(X[ca[cap]])[:, 1]
            te_score[tep] = model.predict_proba(X[te[tep]])[:, 1]
        output[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": ca_score, "test_score": te_score}
        print(f"{spec.name:<35} year={year} train={len(tr):5d}", flush=True)
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
    cols = _features(names)["compact"]
    X = X[:, cols]
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)

    outputs = {}
    for spec in specs():
        outputs[spec.name] = generate(spec, X, y, dates, currencies, reach)

    # Fixed multiscale-window ensembles.  They test robustness to the estimation
    # window rather than optimizing arbitrary weights on later regimes.
    for kind in ("extra", "hist"):
        members = [outputs[f"global_{kind}_window{w}"] for w in (2, 3, 5)]
        outputs[f"global_{kind}_window_equal"] = combine_outputs(
            members, (1 / 3, 1 / 3, 1 / 3), currencies
        )
        outputs[f"global_{kind}_window_short"] = combine_outputs(
            members, (.60, .30, .10), currencies
        )

    with (OUT / "recency_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general_rows = []
    for name, output in outputs.items():
        general_rows.extend(_metric_grid(
            output, y, dates, currencies, benefit, GENERAL_YEARS, name,
        ))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "recency_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "recency_stage1.csv", index=False)

    # Post-shock-specific boosts cannot be selected on 2017--2020 by definition;
    # keep them in the audit table, but exclude them from formal advancement.
    eligible = stage1[~stage1.candidate.str.contains("postshock")].head(8)
    shock_rows = []
    for row in eligible.itertuples(index=False):
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
    shock["robustness"] = shock[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / "recency_stage2_2022_2023.csv", index=False)

    finalists = list(shock.head(4).candidate)
    finalists += [s.name for s in specs() if "postshock" in s.name]
    stage1_by_name = stage1.set_index("candidate")
    final_rows = []
    for name in finalists:
        if name in stage1_by_name.index:
            row = stage1_by_name.loc[name]
        else:
            continue
        result = evaluate(
            outputs[name], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.rate_target), int(row.rolling_window) or None,
            int(row.cooldown_days),
        )
        result.update({"candidate": name,
                       "status": "retrospective; postshock boost is hypothesis-only"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "recency_final_2024_2026_retrospective.csv", index=False)
    (OUT / "recency_protocol.json").write_text(json.dumps({
        "specs": [s.__dict__ for s in specs()],
        "general_years": GENERAL_YEARS, "shock_years": SHOCK_YEARS,
        "final_years": FINAL_YEARS,
        "postshock_boost_selection": "not eligible for formal selection; retrospective sensitivity only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nGENERAL RECENCY")
    print(stage1[["candidate", "frequency", "lift", "forward_benefit_bps",
                  "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nSHOCK RECENCY")
    print(shock[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nRETROSPECTIVE FINAL")
    print(final[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
