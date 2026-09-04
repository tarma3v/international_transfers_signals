"""Distributional path models for the five-publication no-crossing event.

Instead of fitting ``fav_h5`` directly, each model predicts the vector of five
future cumulative log returns.  A joint empirical residual distribution then
turns that path forecast into the probability that every future cumulative
return is non-negative.  Residual vectors stay joint, preserving the strong
within-path dependence that independent horizon classifiers discard.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
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
    window_years: int | None = None
    alpha: float = 100.0
    residual_window_years: int | None = None


def specs() -> list[Spec]:
    # Frozen before results are inspected.
    return [
        Spec("barrier_ridge_expand_a100", "ridge", alpha=100.0),
        Spec("barrier_ridge_window5_a10", "ridge", window_years=5, alpha=10.0),
        Spec("barrier_ridge_window5_a100", "ridge", window_years=5, alpha=100.0),
        Spec("barrier_local_ridge_window5", "ridge", local=True,
             window_years=5, alpha=30.0),
        Spec("barrier_ridge_expand_resid3", "ridge", alpha=100.0,
             residual_window_years=3),
        Spec("barrier_extra_window5", "extra", window_years=5),
        Spec("barrier_hist_window5", "hist", window_years=5),
    ]


def _model(spec: Spec):
    if spec.kind == "ridge":
        return make_pipeline(RobustScaler(), Ridge(alpha=spec.alpha))
    if spec.kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=350, max_depth=7, min_samples_leaf=45,
            max_features=.70, n_jobs=-1, random_state=SEED,
        )
    if spec.kind == "hist":
        # HistGradientBoostingRegressor is scalar-output; a list is fitted below.
        return [HistGradientBoostingRegressor(
            max_iter=180, learning_rate=.035, max_leaf_nodes=9,
            min_samples_leaf=70, l2_regularization=12.0,
            random_state=SEED + step,
        ) for step in range(5)]
    raise KeyError(spec.kind)


def _fit_predict(model, X: np.ndarray, path: np.ndarray, train: np.ndarray,
                 calib: np.ndarray, test: np.ndarray):
    if isinstance(model, list):
        tr_pred = np.column_stack([
            m.fit(X[train], path[train, step]).predict(X[train])
            for step, m in enumerate(model)
        ])
        ca_pred = np.column_stack([m.predict(X[calib]) for m in model])
        te_pred = np.column_stack([m.predict(X[test]) for m in model])
    else:
        model.fit(X[train], path[train])
        tr_pred = model.predict(X[train])
        ca_pred = model.predict(X[calib])
        te_pred = model.predict(X[test])
    return tr_pred, ca_pred, te_pred


def _survival_probability(mean_path: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Empirical orthant probability in bounded-memory batches."""
    if len(residuals) > 1000:
        take = np.linspace(0, len(residuals) - 1, 1000, dtype=int)
        residuals = residuals[take]
    result = np.empty(len(mean_path), dtype=float)
    for start in range(0, len(mean_path), 256):
        values = mean_path[start:start + 256, None, :] + residuals[None, :, :]
        result[start:start + 256] = np.all(values >= 0.0, axis=2).mean(axis=1)
    return result


def _scores_by_currency(
    train: np.ndarray,
    calib: np.ndarray,
    test: np.ndarray,
    tr_pred: np.ndarray,
    ca_pred: np.ndarray,
    te_pred: np.ndarray,
    path: np.ndarray,
    dates: np.ndarray,
    currencies: np.ndarray,
    residual_window_years: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    ca_score = np.full(len(calib), np.nan)
    te_score = np.full(len(test), np.nan)
    for currency in CORRIDORS:
        trm = currencies[train] == currency
        cam = currencies[calib] == currency
        tem = currencies[test] == currency
        residual_mask = trm.copy()
        if residual_window_years is not None and trm.any():
            end = max(dates[train[trm]])
            lower = dt.date(end.year - residual_window_years, end.month, end.day)
            residual_mask &= dates[train] >= lower
        residuals = path[train[residual_mask]] - tr_pred[residual_mask]
        # Sort makes deterministic thinning insensitive to model parallelism.
        residuals = residuals[np.argsort(residuals[:, -1])]
        ca_score[cam] = _survival_probability(ca_pred[cam], residuals)
        te_score[tem] = _survival_probability(te_pred[tem], residuals)
    return ca_score, te_score


def generate(spec: Spec, X: np.ndarray, path: np.ndarray, dates: np.ndarray,
             currencies: np.ndarray, reach: np.ndarray) -> dict:
    output = {}
    for year in ALL_YEARS:
        test_start = dt.date(year, 1, 1)
        calib_start = dt.date(year - 1, 1, 1)
        train_mask = np.asarray([r < calib_start for r in reach]) & np.all(
            np.isfinite(path), axis=1
        )
        if spec.window_years is not None:
            lower = dt.date(year - 1 - spec.window_years, 1, 1)
            train_mask &= dates >= lower
        calib_mask = (dates >= calib_start) & (dates < test_start) & np.all(
            np.isfinite(path), axis=1
        )
        test_mask = np.asarray([d.year == year for d in dates]) & np.all(
            np.isfinite(path), axis=1
        )
        tr, ca, te = np.where(train_mask)[0], np.where(calib_mask)[0], np.where(test_mask)[0]
        ca_score = np.full(len(ca), np.nan)
        te_score = np.full(len(te), np.nan)
        groups = CORRIDORS if spec.local else (None,)
        for currency in groups:
            if currency is None:
                trp, cap, tep = tr, np.arange(len(ca)), np.arange(len(te))
            else:
                trp = tr[currencies[tr] == currency]
                cap = np.where(currencies[ca] == currency)[0]
                tep = np.where(currencies[te] == currency)[0]
            model = _model(spec)
            tr_pred, ca_pred, te_pred = _fit_predict(
                model, X, path, trp, ca[cap], te[tep]
            )
            cs, ts = _scores_by_currency(
                trp, ca[cap], te[tep], tr_pred, ca_pred, te_pred,
                path, dates, currencies, spec.residual_window_years,
            )
            ca_score[cap], te_score[tep] = cs, ts
        output[year] = {
            "calib_idx": ca, "test_idx": te,
            "calib_score": ca_score, "test_score": te_score,
        }
        print(f"{spec.name:<34} year={year} train={len(tr):5d}", flush=True)
    return output


def _future_path(series: dict, index: list[tuple], X: np.ndarray,
                 names: list[str]) -> np.ndarray:
    result = np.full((len(index), 5), np.nan)
    scale = np.maximum(X[:, names.index("raw_vol_20")], 1.0)
    for row, (currency, pos, _day) in enumerate(index):
        values = series[currency].values
        if pos + 5 >= len(values):
            continue
        future = np.log(values[pos + 1:pos + 6] / values[pos]) * 10000.0
        result[row] = np.clip(future / scale[row], -20.0, 20.0)
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
    cols = _features(names)["compact"]
    path = _future_path(series, index, X, names)
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = _benefit(series, index)

    outputs = {}
    for spec in specs():
        outputs[spec.name] = generate(spec, X[:, cols], path, dates, currencies, reach)
    with (OUT / "barrier_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general_rows = []
    for name, output in outputs.items():
        general_rows.extend(_metric_grid(
            output, y, dates, currencies, benefit, GENERAL_YEARS, name
        ))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "barrier_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[
        ["lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "barrier_stage1.csv", index=False)

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
    shock.to_csv(OUT / "barrier_stage2_2022_2023.csv", index=False)

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
    final.to_csv(OUT / "barrier_final_2024_2026_retrospective.csv", index=False)

    (OUT / "barrier_protocol.json").write_text(json.dumps({
        "specs": [s.__dict__ for s in specs()],
        "path_target": "five cumulative future log returns divided by causal raw_vol_20",
        "score": "joint empirical residual probability that all five path values are >= 0",
        "residuals": "currency-specific and training-only",
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

