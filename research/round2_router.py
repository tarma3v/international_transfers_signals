"""Strictly out-of-fold, causal mixture-of-experts experiment.

This implements the error-routing idea separately from the base-model search.
For a forecast year Y the router is trained only on test-fold predictions from
years strictly earlier than Y-1.  It therefore cannot see either the current
test year or its preceding calibration year.  The state variables used by the
router are all observable on the signal date.

Expert scores are converted to within-currency percentile ranks against each
expert's preceding calibration fold.  This makes heterogeneous classifier,
quantile and survival scores comparable without using test labels.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_diverse_models import (
    FINAL_YEARS, SHOCK_YEARS, _metric_grid, _panel_features, _select,
)

OUT = Path("results/research/round2")
SEED = 20260904
GENERAL_YEARS = (2019, 2020)

# Frozen for the router before its results are inspected.  The set deliberately
# spans global/local, parametric/non-parametric, direct/survival/floor models.
EXPERTS = (
    "global_compact_extra",
    "global_path_knn300",
    "local_spline_logit",
    "global_gmm3_hist",
    "local_floor_q35",
    "global_survival_logit",
)

STATE_FEATURES = (
    "ret_5", "ret_20", "ret_60", "range_pos_20", "range_pos_60",
    "slope_z_20", "slope_z_60", "raw_vol_20", "raw_vol_60",
    "vol_ratio_20_120", "positive_share_20", "ret_ac1_20",
    "common_ret_5", "common_ret_20", "common_ret_60",
    "common_ret1_vol20", "residual_ret_5", "residual_ret_20",
    "peer_dispersion_5", "rel_to_peers_5", "annual_sin_1",
    "annual_cos_1", "dow_sin", "dow_cos", "gap_days",
)


def _rank_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def _ranked_part(part: dict, currencies: np.ndarray) -> dict:
    ranked = {}
    for year, z in part.items():
        ca = np.asarray(z["calib_idx"], dtype=int)
        te = np.asarray(z["test_idx"], dtype=int)
        ca_rank = np.full(len(ca), np.nan)
        te_rank = np.full(len(te), np.nan)
        for currency in CORRIDORS:
            cm = currencies[ca] == currency
            tm = currencies[te] == currency
            ca_rank[cm] = _rank_against(z["calib_score"][cm], z["calib_score"][cm])
            te_rank[tm] = _rank_against(z["calib_score"][cm], z["test_score"][tm])
        ranked[year] = {
            "calib_idx": ca, "test_idx": te,
            "calib_score": ca_rank, "test_score": te_rank,
        }
    return ranked


def _expert_matrix(ranked: dict, year: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    key_idx, key_score = f"{split}_idx", f"{split}_score"
    first = ranked[EXPERTS[0]][year]
    rows = np.asarray(first[key_idx], dtype=int)
    scores = []
    for expert in EXPERTS:
        z = ranked[expert][year]
        if not np.array_equal(rows, z[key_idx]):
            raise ValueError(f"unaligned {split} rows for {expert} in {year}")
        scores.append(np.asarray(z[key_score], dtype=float))
    return rows, np.column_stack(scores)


def _past_oof(ranked: dict, year: int) -> tuple[np.ndarray, np.ndarray]:
    rows, scores = [], []
    available = set.intersection(*(set(ranked[e]) for e in EXPERTS))
    for old_year in sorted(available):
        if old_year >= year - 1:
            continue
        old_rows, old_scores = _expert_matrix(ranked, old_year, "test")
        rows.append(old_rows); scores.append(old_scores)
    if not rows:
        return np.array([], dtype=int), np.empty((0, len(EXPERTS)))
    return np.concatenate(rows), np.vstack(scores)


def _softmax(values: np.ndarray, temperature: float = .05) -> np.ndarray:
    values = np.asarray(values, dtype=float) / temperature
    values -= np.max(values, axis=-1, keepdims=True)
    weights = np.exp(np.clip(values, -40, 40))
    return weights / weights.sum(axis=-1, keepdims=True)


def _utility(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Negative rank-squared error; used only as a relative expert loss."""
    return -np.square(scores - target[:, None])


def _regime_codes(X: np.ndarray, names: list[str], rows: np.ndarray,
                  currencies: np.ndarray) -> np.ndarray:
    def col(name): return X[rows, names.index(name)]
    trend = np.digitize(col("ret_20"), (-75.0, 75.0))
    vol = np.digitize(col("vol_ratio_20_120"), (.70, 1.10))
    position = np.digitize(col("range_pos_60"), (25.0, 75.0))
    common = np.digitize(col("common_ret_20"), (-75.0, 75.0))
    currency = pd.Categorical(currencies[rows], categories=CORRIDORS).codes
    return np.column_stack([currency, trend, vol, position, common])


def _hierarchical_regime_utility(train_codes: np.ndarray, train_utility: np.ndarray,
                                 target_codes: np.ndarray) -> np.ndarray:
    """Empirical regime expert skill with shrinking hierarchical back-offs."""
    global_mean = np.nanmean(train_utility, axis=0)
    predicted = np.tile(global_mean, (len(target_codes), 1))
    # Coarse-to-fine updates.  Each finer estimate is shrunk to the estimate
    # already inherited from its parent cell.
    for columns, shrink in (((0,), 180.0), ((0, 1), 120.0),
                            ((0, 1, 2), 90.0), ((0, 1, 2, 3), 70.0),
                            ((0, 1, 2, 3, 4), 50.0)):
        train_keys = [tuple(row[list(columns)]) for row in train_codes]
        target_keys = [tuple(row[list(columns)]) for row in target_codes]
        table = {}
        for key in set(train_keys):
            mask = np.asarray([value == key for value in train_keys])
            table[key] = (int(mask.sum()), np.nanmean(train_utility[mask], axis=0))
        for key in set(target_keys):
            if key not in table:
                continue
            mask = np.asarray([value == key for value in target_keys])
            n, cell = table[key]
            weight = n / (n + shrink)
            predicted[mask] = weight * cell + (1.0 - weight) * predicted[mask]
    return predicted


def _currency_utility(train_utility: np.ndarray, train_rows: np.ndarray,
                      target_rows: np.ndarray, currencies: np.ndarray) -> np.ndarray:
    global_mean = np.nanmean(train_utility, axis=0)
    result = np.tile(global_mean, (len(target_rows), 1))
    for currency in CORRIDORS:
        train_mask = currencies[train_rows] == currency
        target_mask = currencies[target_rows] == currency
        if not train_mask.any():
            continue
        n = int(train_mask.sum())
        weight = n / (n + 250.0)
        result[target_mask] = (
            weight * np.nanmean(train_utility[train_mask], axis=0)
            + (1.0 - weight) * global_mean
        )
    return result


def _learned_prediction(kind: str, train_x: np.ndarray, train_u: np.ndarray,
                        target_x: np.ndarray) -> np.ndarray:
    if kind == "extra":
        model = ExtraTreesRegressor(
            n_estimators=450, max_depth=7, min_samples_leaf=70,
            max_features=.65, n_jobs=-1, random_state=SEED,
        )
    elif kind == "ridge":
        model = make_pipeline(RobustScaler(), Ridge(alpha=100.0))
    else:
        raise KeyError(kind)
    model.fit(train_x, train_u)
    return model.predict(target_x)


def _combine(mode: str, train_rows: np.ndarray, train_scores: np.ndarray,
             target_rows: np.ndarray, target_scores: np.ndarray,
             X: np.ndarray, names: list[str], currencies: np.ndarray,
             y: np.ndarray, state_cols: np.ndarray) -> np.ndarray:
    if mode == "equal":
        return np.nanmean(target_scores, axis=1)
    train_u = _utility(train_scores, y[train_rows])
    if mode == "global_soft":
        expected = np.tile(np.nanmean(train_u, axis=0), (len(target_rows), 1))
    elif mode == "currency_soft":
        expected = _currency_utility(train_u, train_rows, target_rows, currencies)
    elif mode.startswith("regime_"):
        expected = _hierarchical_regime_utility(
            _regime_codes(X, names, train_rows, currencies), train_u,
            _regime_codes(X, names, target_rows, currencies),
        )
    elif mode.startswith("extra_") or mode.startswith("ridge_"):
        kind = mode.split("_", 1)[0]
        train_x = np.column_stack([X[train_rows][:, state_cols], train_scores])
        target_x = np.column_stack([X[target_rows][:, state_cols], target_scores])
        train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
        target_x = np.nan_to_num(target_x, nan=0.0, posinf=0.0, neginf=0.0)
        expected = _learned_prediction(kind, train_x, train_u, target_x)
    else:
        raise KeyError(mode)
    if mode.endswith("hard"):
        chosen = np.argmax(expected, axis=1)
        return target_scores[np.arange(len(target_scores)), chosen]
    weights = _softmax(expected, temperature=.05)
    return np.sum(weights * target_scores, axis=1)


def build_router(mode: str, ranked: dict, X: np.ndarray, names: list[str],
                 currencies: np.ndarray, y: np.ndarray,
                 state_cols: np.ndarray) -> dict:
    output = {}
    available = sorted(set.intersection(*(set(ranked[e]) for e in EXPERTS)))
    for year in available:
        ca, ca_scores = _expert_matrix(ranked, year, "calib")
        te, te_scores = _expert_matrix(ranked, year, "test")
        train_rows, train_scores = _past_oof(ranked, year)
        if mode != "equal" and len(train_rows) < 700:
            continue
        ca_score = _combine(mode, train_rows, train_scores, ca, ca_scores,
                            X, names, currencies, y, state_cols)
        te_score = _combine(mode, train_rows, train_scores, te, te_scores,
                            X, names, currencies, y, state_cols)
        output[year] = {
            "calib_idx": ca, "test_idx": te,
            "calib_score": ca_score, "test_score": te_score,
        }
        print(f"{mode:<18} year={year} gate_train={len(train_rows):5d}", flush=True)
    return output


def _benefit(series, index) -> np.ndarray:
    result = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, 5)
        if value is not None:
            result[row] = value
    return result


def _error_profiles(ranked: dict, X: np.ndarray, names: list[str], y: np.ndarray,
                    currencies: np.ndarray, years: tuple[int, ...], period: str) -> pd.DataFrame:
    frames = []
    for expert in EXPERTS:
        for year in years:
            if year not in ranked[expert]:
                continue
            rows = np.asarray(ranked[expert][year]["test_idx"], dtype=int)
            scores = np.asarray(ranked[expert][year]["test_score"], dtype=float)
            codes = _regime_codes(X, names, np.asarray(rows), currencies)
            frame = pd.DataFrame(codes, columns=["currency_code", "trend", "vol", "position", "common"])
            frame["currency"] = currencies[rows]
            frame["target"] = y[rows]
            frame["score_rank"] = scores
            frame["rank_squared_error"] = np.square(frame.score_rank - frame.target)
            frame["expert"] = expert; frame["year"] = year; frame["period"] = period
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    group = ["period", "expert", "currency", "trend", "vol", "position", "common"]
    result = raw.groupby(group, as_index=False).agg(
        n=("target", "size"), base_rate=("target", "mean"),
        mean_score_rank=("score_rank", "mean"),
        rank_squared_error=("rank_squared_error", "mean"),
    )
    return result[result.n >= 20]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series = load_or_build()
    X, names = _panel_features(X, names, index)
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = _benefit(series, index)
    state_names = list(STATE_FEATURES) + [n for n in names if n.startswith("currency_")]
    missing = [name for name in state_names if name not in names]
    if missing:
        raise KeyError(f"missing router state features: {missing}")
    state_cols = np.asarray([names.index(name) for name in state_names], dtype=int)

    with (OUT / "diverse_outputs.pkl").open("rb") as fh:
        base = pickle.load(fh)
    ranked = {expert: _ranked_part(base[expert], currencies) for expert in EXPERTS}

    modes = (
        "equal", "global_soft", "currency_soft", "regime_soft", "regime_hard",
        "ridge_soft", "extra_soft", "extra_hard",
    )
    routers = {
        mode: build_router(mode, ranked, X, names, currencies, y, state_cols)
        for mode in modes
    }
    with (OUT / "router_outputs.pkl").open("wb") as fh:
        pickle.dump(routers, fh, protocol=pickle.HIGHEST_PROTOCOL)

    general_rows = []
    for mode, output in routers.items():
        general_rows.extend(_metric_grid(
            output, y, dates, currencies, benefit, GENERAL_YEARS, mode,
        ))
    general = pd.DataFrame(general_rows)
    general.to_csv(OUT / "router_general_2019_2020.csv", index=False)
    stage1 = pd.DataFrame([_select(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "router_stage1.csv", index=False)

    shock_rows = []
    for row in stage1.head(6).itertuples(index=False):
        result = evaluate(
            routers[row.candidate], y, dates, currencies, benefit, SHOCK_YEARS,
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
    shock.to_csv(OUT / "router_stage2_2022_2023.csv", index=False)

    final_rows = []
    for row in shock.head(3).itertuples(index=False):
        result = evaluate(
            routers[row.candidate], y, dates, currencies, benefit, FINAL_YEARS,
            float(row.stage1_rate), int(row.stage1_rolling) or None,
            int(row.stage1_cooldown),
        )
        result.update({"candidate": row.candidate,
                       "status": "retrospective: final block previously inspected"})
        final_rows.append(result)
    final = pd.DataFrame(final_rows).sort_values("lift", ascending=False)
    final.to_csv(OUT / "router_final_2024_2026_retrospective.csv", index=False)

    _error_profiles(ranked, X, names, y, currencies, GENERAL_YEARS, "general").to_csv(
        OUT / "router_error_profiles_general.csv", index=False
    )
    _error_profiles(ranked, X, names, y, currencies, SHOCK_YEARS, "shock").to_csv(
        OUT / "router_error_profiles_shock.csv", index=False
    )
    (OUT / "router_protocol.json").write_text(json.dumps({
        "experts": EXPERTS, "modes": modes, "state_features": state_names,
        "general_years": GENERAL_YEARS, "shock_years": SHOCK_YEARS,
        "final_years": FINAL_YEARS,
        "gate_train_rule": "expert test OOF years < forecast_year - 1",
        "score_normalization": "within-currency percentile vs preceding calibration fold",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nGENERAL ROUTERS")
    print(stage1[["candidate", "frequency", "lift", "forward_benefit_bps",
                  "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nSHOCK ROUTERS")
    print(shock[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print("\nRETROSPECTIVE FINAL")
    print(final[["candidate", "frequency", "lift", "forward_benefit_bps",
                "year_lift_min", "corridor_lift_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
