"""Round-two local/global residual towers for the h=5 target.

The key experiment is deliberately different from the anchor search:

1. fit a base classifier separately for every currency;
2. retain only genuinely out-of-year base predictions;
3. train a pooled second-stage model on older out-of-year errors;
4. apply the correction to the preceding calibration year and current test year;
5. learn the alert threshold from calibration scores only.

For a test year Y, the residual learner sees base-model errors no later than
Y-2.  Thus the same fitted residual model scores both calibration year Y-1 and
test year Y, avoiding an in-sample calibration distribution.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from research.extended_features import load_or_build
from research.model_study import evaluate

OUT = Path("results/research/round2")
SEED = 20260904
GENERAL_YEARS = (2019, 2020)
SHOCK_YEARS = (2022, 2023)
FINAL_YEARS = (2024, 2025, 2026)

# Fixed before running a tower.  These describe state rather than reproducing
# the old 90-day anchor formula.  Currency one-hot fields are appended below.
STATE_FEATURES = (
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "ret_120",
    "raw_ret1_lag_1", "raw_ret1_lag_2", "raw_ret1_lag_3",
    "raw_ret1_lag_4", "raw_ret1_lag_5",
    "range_pos_10", "range_pos_20", "range_pos_60", "range_pos_120",
    "range_pos_250", "range_pos_500",
    "slope_z_10", "slope_z_20", "slope_z_60", "slope_z_120", "slope_z_250",
    "raw_vol_5", "raw_vol_20", "raw_vol_60", "raw_vol_120",
    "vol_ratio_5_60", "vol_ratio_20_120",
    "positive_share_5", "positive_share_20", "positive_share_60",
    "ret_ac1_20", "ret_ac1_60", "ret_ac5_20", "ret_ac5_60",
    "bars_since_min_30", "bars_since_max_30", "streak_up", "streak_dn",
    "usd_raw_ret_1", "usd_raw_ret_5", "usd_raw_ret_20", "usd_raw_ret_60",
    "cny_raw_ret_1", "cny_raw_ret_5", "cny_raw_ret_20", "cny_raw_ret_60",
    "eur_raw_ret_1", "eur_raw_ret_5", "eur_raw_ret_20", "eur_raw_ret_60",
    "cnyusd_ret_5", "cnyusd_ret_20", "eurusd_ret_5", "eurusd_ret_20",
    "peer_dispersion_5", "rel_to_peers_5",
    "annual_sin_1", "annual_cos_1", "annual_sin_2", "annual_cos_2",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "gap_days",
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _model(kind: str):
    if kind == "ridge_residual":
        return make_pipeline(RobustScaler(), Ridge(alpha=80.0))
    if kind == "hist_residual":
        return HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.035, max_leaf_nodes=7,
            min_samples_leaf=80, l2_regularization=12.0, random_state=SEED,
        )
    if kind == "extra_residual":
        return ExtraTreesRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=45,
            max_features=0.65, n_jobs=-1, random_state=SEED,
        )
    if kind == "hist_stack":
        return HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.035, max_leaf_nodes=7,
            min_samples_leaf=80, l2_regularization=12.0, random_state=SEED,
        )
    if kind == "xgb_offset":
        return XGBClassifier(
            n_estimators=280, max_depth=2, learning_rate=0.025,
            min_child_weight=70, subsample=0.80, colsample_bytree=0.70,
            reg_lambda=15.0, reg_alpha=1.0, n_jobs=-1,
            random_state=SEED, eval_metric="logloss",
        )
    raise KeyError(kind)


def _past_oof(part: dict, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Return row ids and base scores ending no later than year-2."""
    rows, scores = [], []
    for old_year in sorted(part):
        if old_year >= year - 1:
            continue
        rows.append(np.asarray(part[old_year]["test_idx"], dtype=int))
        scores.append(np.asarray(part[old_year]["test_score"], dtype=float))
    if not rows:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(rows), np.concatenate(scores)


def build_tower(
    part: dict,
    X: np.ndarray,
    y: np.ndarray,
    currencies: np.ndarray,
    cols: np.ndarray,
    kind: str,
    local_correction: bool = False,
) -> dict:
    """Generate strictly chronological second-stage scores."""
    output = {}
    for year in sorted(part):
        train_rows, train_base = _past_oof(part, year)
        if len(train_rows) < 500:
            continue
        z = part[year]
        ca = np.asarray(z["calib_idx"], dtype=int)
        te = np.asarray(z["test_idx"], dtype=int)
        ca_base = np.asarray(z["calib_score"], dtype=float)
        te_base = np.asarray(z["test_score"], dtype=float)
        train_x = np.column_stack([X[train_rows][:, cols], train_base])
        ca_x = np.column_stack([X[ca][:, cols], ca_base])
        te_x = np.column_stack([X[te][:, cols], te_base])
        ca_score = np.full(len(ca), np.nan)
        te_score = np.full(len(te), np.nan)

        groups = CORRIDORS if local_correction else (None,)
        for currency in groups:
            if currency is None:
                tr_pos = np.arange(len(train_rows))
                ca_pos = np.arange(len(ca))
                te_pos = np.arange(len(te))
            else:
                tr_pos = np.where(currencies[train_rows] == currency)[0]
                ca_pos = np.where(currencies[ca] == currency)[0]
                te_pos = np.where(currencies[te] == currency)[0]
            if min(len(tr_pos), len(ca_pos), len(te_pos)) == 0:
                continue
            model = _model(kind)
            if kind == "xgb_offset":
                model.fit(
                    train_x[tr_pos, :-1], y[train_rows[tr_pos]],
                    base_margin=_logit(train_base[tr_pos]),
                )
                ca_score[ca_pos] = model.predict_proba(
                    ca_x[ca_pos, :-1], base_margin=_logit(ca_base[ca_pos])
                )[:, 1]
                te_score[te_pos] = model.predict_proba(
                    te_x[te_pos, :-1], base_margin=_logit(te_base[te_pos])
                )[:, 1]
            elif kind.endswith("residual"):
                model.fit(train_x[tr_pos], y[train_rows[tr_pos]] - train_base[tr_pos])
                ca_score[ca_pos] = np.clip(
                    ca_base[ca_pos] + model.predict(ca_x[ca_pos]), 0.0, 1.0
                )
                te_score[te_pos] = np.clip(
                    te_base[te_pos] + model.predict(te_x[te_pos]), 0.0, 1.0
                )
            else:
                model.fit(train_x[tr_pos], y[train_rows[tr_pos]])
                ca_score[ca_pos] = model.predict_proba(ca_x[ca_pos])[:, 1]
                te_score[te_pos] = model.predict_proba(te_x[te_pos])[:, 1]
        if np.all(np.isfinite(ca_score)) and np.all(np.isfinite(te_score)):
            output[year] = {
                "calib_idx": ca, "test_idx": te,
                "calib_score": ca_score, "test_score": te_score,
            }
    return output


def _benefit(series, index) -> np.ndarray:
    values = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, 5)
        if value is not None:
            values[row] = value
    return values


def _grid(outputs, y, dates, currencies, benefit, years, candidate):
    rows = []
    for rate in (0.20, 0.25, 0.30, 0.35, 0.40):
        for rolling, cooldown in (
            (None, 0), (120, 0), (250, 0), (500, 0), (250, 3), (250, 5)
        ):
            row = evaluate(
                outputs, y, dates, currencies, benefit, years, rate,
                rolling_window=rolling, cooldown_days=cooldown,
            )
            row["candidate"] = candidate
            rows.append(row)
    return rows


def _robustness(frame: pd.DataFrame) -> pd.Series:
    return frame[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series = load_or_build()
    y = build_targets(series, index)["fav_h5"]
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _day in index], dtype=object)
    benefit = _benefit(series, index)
    state = list(STATE_FEATURES) + [n for n in names if n.startswith("currency_")]
    missing = [name for name in state if name not in names]
    if missing:
        raise KeyError(f"missing frozen state features: {missing}")
    cols = np.asarray([names.index(name) for name in state], dtype=int)

    with (Path("results/research") / "candidate_outputs_h5_v2.pkl").open("rb") as fh:
        base_outputs = pickle.load(fh)

    specifications = []
    # The first two are the user's requested local-base -> global-correction tower.
    for base in ("local_logit_top80_5y", "local_cat_top80_5y"):
        for correction in (
            "ridge_residual", "hist_residual", "extra_residual",
            "hist_stack", "xgb_offset",
        ):
            specifications.append((base, correction, False))
    # Reverse and specialist variants test whether pooling belongs in the first
    # or second stage, rather than assuming the requested direction must win.
    for base in ("logit_top80_expand", "hist_top80_expand", "extra_top80_5y"):
        for correction in ("ridge_residual", "hist_residual", "extra_residual"):
            specifications.append((base, correction, True))

    towers = {}
    registry = []
    for base, correction, local_correction in specifications:
        name = (
            f"{base}__{'local' if local_correction else 'global'}_"
            f"{correction}"
        )
        tower = build_tower(
            base_outputs[base], X, y, currencies, cols, correction,
            local_correction=local_correction,
        )
        towers[name] = tower
        registry.append({
            "candidate": name,
            "base": base,
            "correction": correction,
            "correction_scope": "local" if local_correction else "global",
            "state_features": len(cols),
            "first_scored_year": min(tower) if tower else None,
            "last_scored_year": max(tower) if tower else None,
        })
        print(name, sorted(tower), flush=True)

    pd.DataFrame(registry).to_csv(OUT / "tower_registry.csv", index=False)
    with (OUT / "tower_outputs.pkl").open("wb") as fh:
        pickle.dump(towers, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general = []
    for name, tower in towers.items():
        general.extend(_grid(tower, y, dates, currencies, benefit, GENERAL_YEARS, name))
    general = pd.DataFrame(general)
    general["robustness"] = _robustness(general)
    general.to_csv(OUT / "tower_general_2019_2020.csv", index=False)

    # Stage 1: candidate architecture shortlist on 2019--2020.  Each
    # architecture contributes only its best feasible policy.
    feasible = general[
        general.frequency.between(0.90, 2.10)
        & (general.forward_benefit_bps > 0)
        & (general.corridor_freq_min >= 0.65)
    ]
    pool = feasible if len(feasible) else general
    stage1 = (
        pool.sort_values(["candidate", "robustness", "lift"], ascending=[True, False, False])
        .groupby("candidate", as_index=False).head(1)
        .sort_values(["robustness", "lift"], ascending=False).head(8)
    )
    stage1.to_csv(OUT / "tower_stage1_shortlist.csv", index=False)

    # Stage 2: apply those exact candidate-policy pairs to 2022--2023, then
    # choose a finalist without looking at 2024--2026.
    shock_rows = []
    for row in stage1.itertuples(index=False):
        result = evaluate(
            towers[row.candidate], y, dates, currencies, benefit, SHOCK_YEARS,
            float(row.rate_target), int(row.rolling_window) or None,
            int(row.cooldown_days),
        )
        result.update({
            "candidate": row.candidate,
            "stage1_rate_target": float(row.rate_target),
            "stage1_rolling_window": int(row.rolling_window),
            "stage1_cooldown_days": int(row.cooldown_days),
        })
        shock_rows.append(result)
    shock = pd.DataFrame(shock_rows)
    shock["robustness"] = _robustness(shock)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / "tower_stage2_2022_2023.csv", index=False)

    finalists = shock.head(3)
    final_rows = []
    for row in finalists.itertuples(index=False):
        result = evaluate(
            towers[row.candidate], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.stage1_rate_target), int(row.stage1_rolling_window) or None,
            int(row.stage1_cooldown_days),
        )
        result.update({
            "candidate": row.candidate,
            "selected_on": "architecture/policy 2019-2020; finalist 2022-2023",
            "final_status": "retrospective: 2024-2026 was seen in round one",
        })
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "tower_final_2024_2026_retrospective.csv", index=False)

    summary = {
        "general_years": GENERAL_YEARS,
        "shock_years": SHOCK_YEARS,
        "final_years": FINAL_YEARS,
        "state_features": state,
        "finalists": final.to_dict(orient="records"),
    }
    (OUT / "tower_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSTAGE 1")
    print(stage1[["candidate", "frequency", "lift", "forward_benefit_bps",
                  "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nSTAGE 2")
    print(shock[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nRETROSPECTIVE FINAL")
    print(final[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min"]].to_string(index=False))


if __name__ == "__main__":
    main()

