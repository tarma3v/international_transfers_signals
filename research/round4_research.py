"""Round-four leakage-controlled signal research.

Three isolated tracks are evaluated:
1. a hierarchical state estimator using only information at publication i;
2. a timestamp-aware estimator after rate i+1 has been publicly announced;
3. the secondary h=5 ``window closes`` target using information at i.

The final 2024--2026 slice is retrospective because it was inspected in prior
rounds. Architecture and policy selection happen on the earlier blocks.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from research.round2_statistical_audit import _bootstrap_all, _circular_shift_audit, _fired


OUT = Path("results/research/round4")
SEED = 20260904
GENERAL = (2017, 2018, 2019, 2020)
SHOCK = (2022, 2023)
FINAL = (2024, 2025, 2026)
ALL_YEARS = GENERAL + SHOCK + FINAL
SHOCK_DATE = dt.date(2022, 2, 24)
RATES = (.18, .20, .22, .25, .30, .35, .40)
POLICIES = ((None, 0), (120, 0), (250, 0), (500, 0), (250, 3))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], object] | None = None
    window_years: int | None = None
    after_publication: bool = False
    anchor: str | None = None


@dataclass(frozen=True)
class EBSpec:
    name: str
    half_life_years: float | None
    prior: float
    lcb_z: float
    anchor_blend: float


def _logit():
    return make_pipeline(
        RobustScaler(),
        LogisticRegression(C=.06, max_iter=3000, random_state=SEED),
    )


def _hist():
    return HistGradientBoostingClassifier(
        max_iter=260, learning_rate=.035, max_leaf_nodes=11,
        min_samples_leaf=65, l2_regularization=10.0, random_state=SEED,
    )


def _extra():
    return ExtraTreesClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=35,
        max_features=.70, n_jobs=-1, random_state=SEED,
    )


def _xgb():
    return XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=.035,
        min_child_weight=35, subsample=.82, colsample_bytree=.78,
        reg_lambda=8.0, reg_alpha=.35, n_jobs=-1,
        random_state=SEED, eval_metric="logloss",
    )


def _current_columns(names: list[str]) -> np.ndarray:
    requested = [
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
        "range_pos_20", "range_pos_60", "range_pos_120", "range_pos_250",
        "rank_level_20", "rank_level_60", "rank_level_120", "rank_level_250",
        "slope_z_20", "slope_z_60", "slope_z_120",
        "raw_vol_5", "raw_vol_20", "raw_vol_60", "raw_vol_120",
        "vol_ratio_5_60", "vol_ratio_20_120", "positive_share_20",
        "positive_share_60", "ret_ac1_20", "ret_ac1_60",
        "streak_up", "streak_dn", "bars_since_min_30", "bars_since_max_30",
        "level_vs_ema_5", "level_vs_ema_20", "level_vs_ema_60",
        "macd_5_20", "macd_20_60", "peer_dispersion_5", "rel_to_peers_5",
        "usd_ret_5", "usd_ret_20", "cny_ret_5", "cny_ret_20",
        "usd_raw_ret_5", "usd_raw_ret_20", "cny_raw_ret_5", "cny_raw_ret_20",
        "eur_raw_ret_5", "eur_raw_ret_20", "eurusd_ret_20", "cnyusd_ret_20",
        "annual_sin_1", "annual_cos_1", "annual_sin_2", "annual_cos_2",
        "dow_sin", "dow_cos", "gap_days", "high_vol_regime", "quiet_regime",
        "first_week_month", "last_week_month", "pre_new_year_14",
    ]
    requested += [name for name in names if name.startswith("currency_")]
    missing = [name for name in requested if name not in names]
    if missing:
        print(f"Skipping unavailable compact features: {missing}", flush=True)
    return np.asarray([names.index(name) for name in requested if name in names], dtype=int)


def _future_benefit(series, index, h=5) -> np.ndarray:
    out = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, h)
        if value is not None:
            out[row] = value
    return out


def _masks(year, dates, reach, y, window_years=None):
    test_start = dt.date(year, 1, 1)
    calib_start = dt.date(year - 1, 1, 1)
    train = np.asarray([r < calib_start for r in reach]) & ~np.isnan(y)
    if window_years is not None:
        lower = dt.date(year - 1 - window_years, 1, 1)
        train &= np.asarray([day >= lower for day in dates])
    calib = np.asarray([calib_start <= day < test_start for day in dates]) & ~np.isnan(y)
    test = np.asarray([day.year == year for day in dates]) & ~np.isnan(y)
    return np.where(train)[0], np.where(calib)[0], np.where(test)[0]


def _rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def _anchor_score(X, names, kind):
    col = lambda n: X[:, names.index(n)]
    if kind == "upper_range":
        return col("pct_range_90")
    if kind == "trend":
        return col("pct_range_90") + .035 * col("ret_20") + .015 * col("ret_60")
    raise KeyError(kind)


def _generate_models(specs, matrix, y, dates, reach, years, anchors=None, eligibility=None):
    outputs = {}
    for spec in specs:
        per_year = {}
        for year in years:
            tr, ca, te = _masks(year, dates, reach, y, spec.window_years)
            if min(len(tr), len(ca), len(te)) == 0:
                continue
            if eligibility is not None:
                tr_fit = tr[eligibility[tr]]
            else:
                tr_fit = tr
            if spec.anchor:
                score = anchors[spec.anchor]
                cs, ts = score[ca], score[te]
            else:
                model = spec.factory()
                model.fit(matrix[tr_fit], y[tr_fit])
                cs = model.predict_proba(matrix[ca])[:, 1]
                ts = model.predict_proba(matrix[te])[:, 1]
            if eligibility is not None:
                cs = np.where(eligibility[ca], cs, -1.0)
                ts = np.where(eligibility[te], ts, -1.0)
            per_year[year] = {
                "calib_idx": ca, "test_idx": te,
                "calib_score": cs, "test_score": ts,
            }
            print(f"  {spec.name:<30} year={year} train={len(tr_fit):5d}", flush=True)
        outputs[spec.name] = per_year
    return outputs


class HierarchicalStates:
    """Partially pooled weighted target-rate lookup for sparse market states."""

    def __init__(self, prior: float, lcb_z: float, half_life_years: float | None):
        self.prior = prior
        self.lcb_z = lcb_z
        self.half_life_years = half_life_years
        self.edges = {}
        self.tables = []
        self.global_p = .5

    @staticmethod
    def _safe_edges(x):
        edges = np.unique(np.quantile(x, [.2, .4, .6, .8]))
        return edges if len(edges) else np.asarray([0.0])

    def _state(self, X, names, currencies, dates, fit=False):
        get = lambda name: X[:, names.index(name)]
        if fit:
            self.edges = {
                "ret20": self._safe_edges(get("ret_20")),
                "ret60": self._safe_edges(get("ret_60")),
                "vol": self._safe_edges(get("vol_ratio_20_120")),
            }
        rb = np.digitize(get("pct_range_90"), [10, 25, 40, 55, 70, 82, 90, 95])
        r20 = np.digitize(get("ret_20"), self.edges["ret20"])
        r60 = np.digitize(get("ret_60"), self.edges["ret60"])
        vol = np.digitize(get("vol_ratio_20_120"), self.edges["vol"])
        month = np.asarray([day.month for day in dates])
        return [
            list(zip(rb)),
            list(zip(currencies, rb)),
            list(zip(rb, r20)),
            list(zip(rb, r60)),
            list(zip(rb, vol)),
            list(zip(currencies, rb, np.sign(get("ret_20")).astype(int))),
            list(zip(currencies, month)),
        ]

    def fit(self, X, y, names, currencies, dates, cutoff):
        if self.half_life_years:
            age = np.asarray([(cutoff - day).days for day in dates], dtype=float)
            weights = np.power(.5, age / (365.25 * self.half_life_years))
        else:
            weights = np.ones(len(y))
        self.global_p = float(np.sum(weights * y) / np.sum(weights))
        self.tables = []
        for keys in self._state(X, names, currencies, dates, fit=True):
            table = {}
            for key, target, weight in zip(keys, y, weights):
                count, success = table.get(key, (0.0, 0.0))
                table[key] = (count + weight, success + weight * target)
            self.tables.append(table)
        return self

    def predict(self, X, names, currencies, dates):
        groups = self._state(X, names, currencies, dates, fit=False)
        component_weights = np.asarray([.22, .20, .18, .14, .10, .11, .05])
        result = np.zeros(len(X))
        for weight, keys, table in zip(component_weights, groups, self.tables):
            values = np.empty(len(X))
            for j, key in enumerate(keys):
                count, success = table.get(key, (0.0, 0.0))
                p = (success + self.prior * self.global_p) / (count + self.prior)
                se = np.sqrt(max(p * (1.0 - p), 1e-9) / (count + self.prior + 1.0))
                values[j] = p - self.lcb_z * se
            result += weight * values
        return result


class DirectionalMarkovStates(HierarchicalStates):
    """Smoothed target rates for recent up/flat/down transition patterns."""

    def _state(self, X, names, currencies, dates, fit=False):
        get = lambda name: X[:, names.index(name)]
        moves = []
        for lag in range(1, 6):
            raw = get(f"raw_ret1_lag_{lag}")
            moves.append(np.where(raw > 1e-10, 1, np.where(raw < -1e-10, -1, 0)))
        rb = np.digitize(get("pct_range_90"), [10, 25, 40, 55, 70, 82, 90, 95])
        vol = np.digitize(get("vol_ratio_20_120"), [.70, 1.00, 1.35])
        streak = np.clip(
            np.where(get("streak_up") > 0, get("streak_up"), -get("streak_dn")),
            -4, 4,
        ).astype(int)
        month = np.asarray([day.month for day in dates])
        return [
            list(zip(moves[0])),
            list(zip(currencies, moves[0])),
            list(zip(moves[0], moves[1])),
            list(zip(moves[0], moves[1], moves[2])),
            list(zip(currencies, moves[0], moves[1])),
            list(zip(rb, moves[0], moves[1])),
            list(zip(vol, streak)),
            list(zip(currencies, month)),
        ]

    def predict(self, X, names, currencies, dates):
        groups = self._state(X, names, currencies, dates, fit=False)
        component_weights = np.asarray([.12, .15, .16, .12, .15, .15, .10, .05])
        result = np.zeros(len(X))
        for weight, keys, table in zip(component_weights, groups, self.tables):
            values = np.empty(len(X))
            for j, key in enumerate(keys):
                count, success = table.get(key, (0.0, 0.0))
                p = (success + self.prior * self.global_p) / (count + self.prior)
                se = np.sqrt(max(p * (1.0 - p), 1e-9) / (count + self.prior + 1.0))
                values[j] = p - self.lcb_z * se
            result += weight * values
        return result


def _eb_specs():
    return [
        EBSpec("eb_expand_p30", None, 30.0, 0.0, 0.0),
        EBSpec("eb_expand_p70_lcb", None, 70.0, .5, 0.0),
        EBSpec("eb_decay4_p40", 4.0, 40.0, 0.0, 0.0),
        EBSpec("eb_decay2_p50_lcb", 2.0, 50.0, .5, 0.0),
        EBSpec("eb_decay4_anchor25", 4.0, 40.0, 0.0, .25),
        EBSpec("eb_decay4_anchor50", 4.0, 40.0, 0.0, .50),
        EBSpec("eb_expand_anchor50", None, 30.0, 0.0, .50),
    ]


def _generate_eb(specs, X, names, y, dates, currencies, reach, years):
    outputs = {}
    anchor = X[:, names.index("pct_range_90")] / 100.0
    for spec in specs:
        per_year = {}
        for year in years:
            tr, ca, te = _masks(year, dates, reach, y)
            cutoff = dt.date(year - 1, 1, 1)
            model = HierarchicalStates(spec.prior, spec.lcb_z, spec.half_life_years)
            model.fit(X[tr], y[tr], names, currencies[tr], dates[tr], cutoff)
            cs = model.predict(X[ca], names, currencies[ca], dates[ca])
            ts = model.predict(X[te], names, currencies[te], dates[te])
            if spec.anchor_blend:
                raw_cs = cs.copy()
                cs = ((1 - spec.anchor_blend) * _rank(raw_cs, raw_cs)
                      + spec.anchor_blend * _rank(anchor[ca], anchor[ca]))
                ts = ((1 - spec.anchor_blend) * _rank(raw_cs, ts)
                      + spec.anchor_blend * _rank(anchor[ca], anchor[te]))
            per_year[year] = {
                "calib_idx": ca, "test_idx": te,
                "calib_score": cs, "test_score": ts,
            }
            print(f"  {spec.name:<30} year={year} train={len(tr):5d}", flush=True)
        outputs[spec.name] = per_year
    return outputs


def _generate_markov(specs, X, names, y, dates, currencies, reach, years):
    outputs = {}
    anchor = X[:, names.index("pct_range_90")] / 100.0
    for spec in specs:
        per_year = {}
        for year in years:
            tr, ca, te = _masks(year, dates, reach, y)
            cutoff = dt.date(year - 1, 1, 1)
            model = DirectionalMarkovStates(spec.prior, spec.lcb_z, spec.half_life_years)
            model.fit(X[tr], y[tr], names, currencies[tr], dates[tr], cutoff)
            raw_cs = model.predict(X[ca], names, currencies[ca], dates[ca])
            raw_ts = model.predict(X[te], names, currencies[te], dates[te])
            if spec.anchor_blend:
                cs = ((1 - spec.anchor_blend) * _rank(raw_cs, raw_cs)
                      + spec.anchor_blend * _rank(anchor[ca], anchor[ca]))
                ts = ((1 - spec.anchor_blend) * _rank(raw_cs, raw_ts)
                      + spec.anchor_blend * _rank(anchor[ca], anchor[te]))
            else:
                cs, ts = raw_cs, raw_ts
            per_year[year] = {
                "calib_idx": ca, "test_idx": te,
                "calib_score": cs, "test_score": ts,
            }
            print(f"  {spec.name:<30} year={year} train={len(tr):5d}", flush=True)
        outputs[spec.name] = per_year
    return outputs


def _generate_postshock_lookup(specs, model_class, X, names, y, dates, currencies, reach):
    """Reset models: train only after 2022-02-24, starting with the 2024 fold."""
    outputs = {}
    anchor = X[:, names.index("pct_range_90")] / 100.0
    for spec in specs:
        per_year = {}
        for year in FINAL:
            tr, ca, te = _masks(year, dates, reach, y)
            tr = tr[np.asarray([dates[row] >= SHOCK_DATE for row in tr])]
            cutoff = dt.date(year - 1, 1, 1)
            model = model_class(spec.prior, spec.lcb_z, spec.half_life_years)
            model.fit(X[tr], y[tr], names, currencies[tr], dates[tr], cutoff)
            raw_cs = model.predict(X[ca], names, currencies[ca], dates[ca])
            raw_ts = model.predict(X[te], names, currencies[te], dates[te])
            if spec.anchor_blend:
                cs = ((1 - spec.anchor_blend) * _rank(raw_cs, raw_cs)
                      + spec.anchor_blend * _rank(anchor[ca], anchor[ca]))
                ts = ((1 - spec.anchor_blend) * _rank(raw_cs, raw_ts)
                      + spec.anchor_blend * _rank(anchor[ca], anchor[te]))
            else:
                cs, ts = raw_cs, raw_ts
            per_year[year] = {
                "calib_idx": ca, "test_idx": te,
                "calib_score": cs, "test_score": ts,
            }
            print(f"  {spec.name:<30} year={year} reset_train={len(tr):5d}", flush=True)
        outputs[spec.name] = per_year
    return outputs


def _combine_outputs(named_outputs, currencies, weights):
    names = list(named_outputs)
    result = {}
    years = sorted(set.intersection(*(set(named_outputs[name]) for name in names)))
    for year in years:
        first = named_outputs[names[0]][year]
        ca, te = first["calib_idx"], first["test_idx"]
        cs = np.zeros(len(ca)); ts = np.zeros(len(te))
        for name, weight in zip(names, weights):
            part = named_outputs[name][year]
            for currency in CORRIDORS:
                cm = currencies[ca] == currency
                tm = currencies[te] == currency
                cs[cm] += weight * _rank(part["calib_score"][cm], part["calib_score"][cm])
                ts[tm] += weight * _rank(part["calib_score"][cm], part["test_score"][tm])
        result[year] = {
            "calib_idx": ca, "test_idx": te,
            "calib_score": cs, "test_score": ts,
        }
    return result


def _publication_matrix(X, names, index, series, cols):
    row_of = {(currency, i): row for row, (currency, i, _day) in enumerate(index)}
    next_rows = np.asarray([row_of.get((currency, i + 1), -1) for currency, i, _day in index])
    eligibility = np.zeros(len(index), dtype=bool)
    known = np.zeros((len(index), 8), dtype=float)
    next_base = np.zeros((len(index), len(cols)), dtype=float)
    by_date = {}
    for row, (currency, i, _day) in enumerate(index):
        if i + 1 >= len(series[currency].values) or next_rows[row] < 0:
            continue
        current = float(series[currency].values[i])
        nxt = float(series[currency].values[i + 1])
        margin = (nxt / current - 1.0) * 10000.0
        eligibility[row] = nxt >= current
        nr = next_rows[row]
        next_base[row] = X[nr, cols]
        day = series[currency].dates[i + 1]
        by_date.setdefault(day, []).append((row, margin))
        vol = max(float(X[nr, names.index("raw_vol_20")]), 1e-6)
        known[row, :4] = [margin, abs(margin), margin / vol, float(margin == 0)]
    for _day, pairs in by_date.items():
        margins = np.asarray([margin for _row, margin in pairs])
        for row, margin in pairs:
            known[row, 4:] = [float(np.mean(margins)), float(np.std(margins)),
                              float(np.min(margins)), float(np.mean(margins <= margin))]
    matrix = np.column_stack([X[:, cols], next_base, known])
    feature_names = (
        [f"current_{names[j]}" for j in cols]
        + [f"published_next_{names[j]}" for j in cols]
        + ["known_margin_bps", "known_abs_margin_bps", "known_margin_vol",
           "known_unchanged", "known_peer_mean", "known_peer_std",
           "known_peer_min", "known_peer_rank"]
    )
    return matrix, feature_names, eligibility, next_rows


def _publication_anchors(X, names, index, series, eligibility, next_rows):
    margin = np.full(len(index), -1.0)
    next_range = np.full(len(index), -1.0)
    next_trend = np.full(len(index), -1.0)
    current_range = X[:, names.index("pct_range_90")]
    for row, (currency, i, _day) in enumerate(index):
        if not eligibility[row]:
            continue
        nr = next_rows[row]
        raw_margin = (series[currency].values[i + 1] / series[currency].values[i] - 1.0) * 10000.0
        margin[row] = raw_margin
        next_range[row] = X[nr, names.index("pct_range_90")]
        next_trend[row] = (
            next_range[row] + .035 * X[nr, names.index("ret_20")]
            + .015 * X[nr, names.index("ret_60")]
        )
    return {
        "known_next_gate": np.where(eligibility, 1.0, -1.0),
        "known_margin": margin,
        "known_next_range": next_range,
        "known_next_trend": next_trend,
        "known_next_blend": np.where(
            eligibility, .45 * current_range + .55 * next_range + .02 * margin, -1.0
        ),
    }


def _grid(outputs, y, dates, currencies, benefit, years, candidates):
    rows = []
    for candidate in candidates:
        for rate in RATES:
            for rolling, cooldown in POLICIES:
                metric = evaluate(
                    outputs[candidate], y, dates, currencies, benefit, years,
                    rate, rolling, cooldown,
                )
                metric["candidate"] = candidate
                rows.append(metric)
    return pd.DataFrame(rows)


def _working_point(frame):
    feasible = frame[
        frame.frequency.between(1.0, 2.0)
        & (frame.corridor_freq_min >= .65)
        & (frame.corridor_freq_max <= 2.50)
        & (frame.forward_benefit_bps > 0)
    ].copy()
    pool = feasible if len(feasible) else frame.copy()
    pool["robustness"] = pool[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    return pool.sort_values(
        ["robustness", "lift", "forward_benefit_bps", "auc"], ascending=False
    ).iloc[0]


def _selection_track(track, outputs, y, dates, currencies, benefit):
    candidates = sorted(outputs)
    general = _grid(outputs, y, dates, currencies, benefit, GENERAL, candidates)
    general.to_csv(OUT / f"{track}_general_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_working_point(z) for _name, z in general.groupby("candidate")])
    stage1["robustness"] = stage1[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / f"{track}_stage1.csv", index=False)

    shock_rows = []
    for row in stage1.itertuples(index=False):
        metric = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, SHOCK,
            float(row.rate_target), int(row.rolling_window) or None, int(row.cooldown_days),
        )
        metric.update({
            "candidate": row.candidate, "selected_rate": row.rate_target,
            "selected_rolling": row.rolling_window,
            "selected_cooldown": row.cooldown_days,
        })
        shock_rows.append(metric)
    shock = pd.DataFrame(shock_rows)
    shock["robustness"] = shock[["lift", "year_lift_min", "corridor_lift_min"]].min(axis=1)
    shock = shock.sort_values(["robustness", "lift"], ascending=False)
    shock.to_csv(OUT / f"{track}_stage2_2022_2023.csv", index=False)

    # Complete retrospective table for diagnosis/ablations. It is explicitly
    # not the shortlist and must not be used to choose a headline candidate.
    all_final_rows = []
    for row in shock.itertuples(index=False):
        metric = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, FINAL,
            float(row.selected_rate), int(row.selected_rolling) or None,
            int(row.selected_cooldown),
        )
        metric.update({
            "candidate": row.candidate, "selected_rate": row.selected_rate,
            "selected_rolling": row.selected_rolling,
            "selected_cooldown": row.selected_cooldown,
            "selection_status": "diagnostic all-candidate retrospective table",
        })
        all_final_rows.append(metric)
    pd.DataFrame(all_final_rows).sort_values("lift", ascending=False).to_csv(
        OUT / f"{track}_all_candidates_final_sensitivity.csv", index=False
    )

    finalists = shock[
        shock.frequency.between(1.0, 2.0)
        & (shock.corridor_freq_min >= .65)
        & (shock.forward_benefit_bps > 0)
    ]
    if not len(finalists):
        finalists = shock
    final_rows = []
    for selection_rank, row in enumerate(finalists.head(4).itertuples(index=False), start=1):
        metric = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, FINAL,
            float(row.selected_rate), int(row.selected_rolling) or None,
            int(row.selected_cooldown),
        )
        metric.update({
            "candidate": row.candidate, "selected_rate": row.selected_rate,
            "selected_rolling": row.selected_rolling,
            "selected_cooldown": row.selected_cooldown,
            "selection_rank": selection_rank,
            "selection_status": "selected on 2017-2023; 2024-2026 retrospective",
        })
        final_rows.append(metric)
    final = pd.DataFrame(final_rows).sort_values("selection_rank")
    final.to_csv(OUT / f"{track}_final_2024_2026_retrospective.csv", index=False)
    return general, stage1, shock, final


def _postshock_selection(outputs, y, dates, currencies, benefit):
    """2024 screen -> 2025 gate -> untouched-by-this-round 2026 audit."""
    candidates = sorted(outputs)
    screening = _grid(outputs, y, dates, currencies, benefit, (2024,), candidates)
    screening.to_csv(OUT / "postshock_states_screen_2024_retrospective.csv", index=False)
    stage1 = pd.DataFrame([_working_point(z) for _name, z in screening.groupby("candidate")])
    stage1["robustness"] = stage1[["lift", "corridor_lift_min"]].min(axis=1)
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "postshock_states_stage1.csv", index=False)

    confirm_rows = []
    for row in stage1.itertuples(index=False):
        metric = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, (2025,),
            float(row.rate_target), int(row.rolling_window) or None, int(row.cooldown_days),
        )
        metric.update({
            "candidate": row.candidate, "selected_rate": row.rate_target,
            "selected_rolling": row.rolling_window,
            "selected_cooldown": row.cooldown_days,
        })
        confirm_rows.append(metric)
    confirm = pd.DataFrame(confirm_rows)
    confirm["robustness"] = confirm[["lift", "corridor_lift_min"]].min(axis=1)
    confirm = confirm.sort_values(["robustness", "lift"], ascending=False)
    confirm.to_csv(OUT / "postshock_states_confirm_2025.csv", index=False)

    feasible = confirm[
        confirm.frequency.between(1.0, 2.0)
        & (confirm.corridor_freq_min >= .65)
        & (confirm.forward_benefit_bps > 0)
    ]
    if not len(feasible):
        feasible = confirm
    audit_rows, combined_rows = [], []
    for row in feasible.head(4).itertuples(index=False):
        settings = (float(row.selected_rate), int(row.selected_rolling) or None,
                    int(row.selected_cooldown))
        audit = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, (2026,), *settings
        )
        audit.update({"candidate": row.candidate, "selected_rate": settings[0],
                      "selected_rolling": settings[1] or 0,
                      "selected_cooldown": settings[2]})
        audit_rows.append(audit)
        combined = evaluate(
            outputs[row.candidate], y, dates, currencies, benefit, FINAL, *settings
        )
        combined.update({"candidate": row.candidate, "selected_rate": settings[0],
                         "selected_rolling": settings[1] or 0,
                         "selected_cooldown": settings[2],
                         "selection_status": "2024 retrospective screen; 2025 confirmation; 2026 audit"})
        combined_rows.append(combined)
    audit = pd.DataFrame(audit_rows).sort_values("lift", ascending=False)
    combined = pd.DataFrame(combined_rows).sort_values("lift", ascending=False)
    audit.to_csv(OUT / "postshock_states_audit_2026.csv", index=False)
    combined.to_csv(OUT / "postshock_states_combined_2024_2026.csv", index=False)
    return stage1, confirm, audit, combined


def _cross_period_robustness(track, outputs, y, dates, currencies, benefit):
    """Multiplicity-labelled diagnostic: same setting across all three eras."""
    general = _grid(outputs, y, dates, currencies, benefit, GENERAL, sorted(outputs))
    rows = []
    for row in general.itertuples(index=False):
        settings = (float(row.rate_target), int(row.rolling_window) or None,
                    int(row.cooldown_days))
        shock = evaluate(outputs[row.candidate], y, dates, currencies, benefit, SHOCK, *settings)
        final = evaluate(outputs[row.candidate], y, dates, currencies, benefit, FINAL, *settings)
        record = {
            "candidate": row.candidate, "rate": settings[0],
            "rolling": settings[1] or 0, "cooldown": settings[2],
            "general_lift": row.lift, "shock_lift": shock["lift"],
            "final_lift": final["lift"],
            "general_frequency": row.frequency, "shock_frequency": shock["frequency"],
            "final_frequency": final["frequency"],
            "general_corridor_lift_min": row.corridor_lift_min,
            "shock_corridor_lift_min": shock["corridor_lift_min"],
            "final_corridor_lift_min": final["corridor_lift_min"],
            "general_benefit": row.forward_benefit_bps,
            "shock_benefit": shock["forward_benefit_bps"],
            "final_benefit": final["forward_benefit_bps"],
            "status": "diagnostic grid, not an unbiased selected estimate",
        }
        record["lift_min_era"] = min(record["general_lift"], record["shock_lift"], record["final_lift"])
        record["frequency_in_band_all_eras"] = all(
            1.0 <= record[name] <= 2.0
            for name in ("general_frequency", "shock_frequency", "final_frequency")
        )
        rows.append(record)
    result = pd.DataFrame(rows).sort_values(
        ["frequency_in_band_all_eras", "lift_min_era"], ascending=False
    )
    result.to_csv(OUT / f"{track}_cross_period_robustness_diagnostic.csv", index=False)
    return result


def _annual_selected(track, outputs, final, y, dates, currencies, benefit):
    selected = final.sort_values("selection_rank").iloc[0]
    settings = (float(selected.selected_rate), int(selected.selected_rolling) or None,
                int(selected.selected_cooldown))
    rows = []
    for year in ALL_YEARS:
        metric = evaluate(
            outputs[selected.candidate], y, dates, currencies, benefit, (year,), *settings
        )
        metric.update({"track": track, "candidate": selected.candidate, "year": year})
        rows.append(metric)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"{track}_selected_annual.csv", index=False)
    return result


def _bootstrap(track, outputs, final, y, benefit, dates, currencies):
    years = FINAL
    valid_ref = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
    policies = {}
    for row in final.itertuples(index=False):
        valid, fired = _fired(
            outputs[row.candidate], years, dates, currencies, y,
            float(row.selected_rate), int(row.selected_rolling) or None,
            int(row.selected_cooldown),
        )
        if np.array_equal(valid, valid_ref):
            policies[row.candidate] = fired
    if not policies:
        return pd.DataFrame()
    draws = _bootstrap_all(y, benefit, dates, valid_ref, policies)
    rows = []
    for name, fired in policies.items():
        active = valid_ref & fired
        d = draws[name]
        rows.append({
            "track": track, "candidate": name, "n": int(active.sum()),
            "lift": float(y[active].mean() / y[valid_ref].mean()),
            "lift_ci_low": float(np.quantile(d["lift"], .025)),
            "lift_ci_high": float(np.quantile(d["lift"], .975)),
            "p_lift_le_1": float((np.sum(d["lift"] <= 1) + 1) / (len(d["lift"]) + 1)),
            "forward_benefit_bps": float(np.nanmean(benefit[active])),
            "benefit_ci_low": float(np.quantile(d["benefit"], .025)),
            "benefit_ci_high": float(np.quantile(d["benefit"], .975)),
        })
    result = pd.DataFrame(rows).sort_values("lift", ascending=False)
    result.to_csv(OUT / f"{track}_bootstrap.csv", index=False)
    return result


def _multiplicity(track, outputs, stage2, y, dates, currencies):
    valid_ref = np.asarray([day.year in FINAL for day in dates]) & ~np.isnan(y)
    policies = {}
    for row in stage2.itertuples(index=False):
        valid, fired = _fired(
            outputs[row.candidate], FINAL, dates, currencies, y,
            float(row.selected_rate), int(row.selected_rolling) or None,
            int(row.selected_cooldown),
        )
        if np.array_equal(valid, valid_ref) and fired.any():
            policies[row.candidate] = fired
    result = _circular_shift_audit(
        y, dates, currencies, valid_ref, policies,
        f"{track}_retrospective_2024_2026",
    )
    result.to_csv(OUT / f"{track}_circular_shift_multiplicity.csv", index=False)
    return result


def _run_track(track, outputs, y, dates, currencies, benefit):
    with (OUT / f"{track}_outputs.pkl").open("wb") as fh:
        pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    general, stage1, shock, final = _selection_track(
        track, outputs, y, dates, currencies, benefit
    )
    boot = _bootstrap(track, outputs, final, y, benefit, dates, currencies)
    multiplicity = _multiplicity(track, outputs, shock, y, dates, currencies)
    print(f"\n{track.upper()} — GENERAL", flush=True)
    print(stage1[["candidate", "frequency", "lift", "forward_benefit_bps",
                  "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print(f"\n{track.upper()} — SHOCK", flush=True)
    print(shock[["candidate", "frequency", "lift", "forward_benefit_bps",
                 "year_lift_min", "corridor_lift_min", "robustness"]].to_string(index=False))
    print(f"\n{track.upper()} — RETROSPECTIVE FINAL", flush=True)
    print(final[["candidate", "frequency", "lift", "forward_benefit_bps",
                 "year_lift_min", "corridor_lift_min"]].to_string(index=False))
    return {"general": general, "stage1": stage1, "shock": shock, "final": final,
            "bootstrap": boot, "multiplicity": multiplicity}


def _write_leakage_audit(index, series, reach, targets, eligibility, next_rows):
    mapping_ok = True
    margin_gate_ok = True
    for row, (currency, i, _day) in enumerate(index):
        if next_rows[row] >= 0:
            nc, ni, _nd = index[next_rows[row]]
            mapping_ok &= nc == currency and ni == i + 1
        if not np.isnan(targets["fav_h1"][row]):
            margin_gate_ok &= eligibility[row] == bool(targets["fav_h1"][row])
    purge_checks = {}
    dates = np.asarray([day for _currency, _i, day in index], dtype=object)
    for year in ALL_YEARS:
        tr, _ca, _te = _masks(year, dates, reach, targets["fav_h5"])
        boundary = dt.date(year - 1, 1, 1)
        purge_checks[str(year)] = bool(all(reach[row] < boundary for row in tr))
    audit = {
        "next_row_is_same_currency_i_plus_1": bool(mapping_ok),
        "known_gate_equals_fav_h1": bool(margin_gate_ok),
        "purged_training_before_calibration": purge_checks,
        "ordinary_matrix_uses_next_row": False,
        "publication_matrix_max_lookahead": "i+1, valid only after that effective rate is public",
        "later_than_i_plus_1_features": False,
    }
    (OUT / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _publication_feature_importance(matrix, feature_names, y, dates, reach, eligibility):
    """Interpret a model fitted before the retrospective final block."""
    tr, _ca, _te = _masks(2024, dates, reach, y, 7)
    tr = tr[eligibility[tr]]
    logit = _logit().fit(matrix[tr], y[tr])
    coef = np.abs(logit.named_steps["logisticregression"].coef_[0])
    extra = _extra().fit(matrix[tr], y[tr])
    frame = pd.DataFrame({
        "feature": feature_names,
        "logit_abs_scaled_coefficient": coef,
        "extra_trees_importance": extra.feature_importances_,
    })
    frame["mean_rank"] = (
        frame.logit_abs_scaled_coefficient.rank(pct=True)
        + frame.extra_trees_importance.rank(pct=True)
    ) / 2.0
    frame.sort_values("mean_rank", ascending=False).to_csv(
        OUT / "after_publication_feature_importance.csv", index=False
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series = load_or_build()
    dates = np.asarray([day for _currency, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _day in index], dtype=object)
    targets = build_targets(series, index)
    benefit = _future_benefit(series, index, 5)
    reach = target_reach_dates(index, series, 5)
    cols = _current_columns(names)
    current = X[:, cols]

    # A. Ordinary, pre-publication state family.
    y_fav = targets["fav_h5"]
    ordinary_outputs = _generate_eb(
        _eb_specs(), X, names, y_fav, dates, currencies, reach, ALL_YEARS
    )
    markov_specs = [
        EBSpec("markov_expand_p40", None, 40.0, 0.0, 0.0),
        EBSpec("markov_decay4_p40", 4.0, 40.0, 0.0, 0.0),
        EBSpec("markov_decay2_lcb", 2.0, 50.0, .5, 0.0),
        EBSpec("markov_decay4_anchor25", 4.0, 40.0, 0.0, .25),
        EBSpec("markov_expand_anchor50", None, 40.0, 0.0, .50),
    ]
    ordinary_outputs.update(_generate_markov(
        markov_specs, X, names, y_fav, dates, currencies, reach, ALL_YEARS
    ))
    ordinary = _run_track(
        "ordinary_states", ordinary_outputs, y_fav, dates, currencies, benefit
    )
    _annual_selected(
        "ordinary_states", ordinary_outputs, ordinary["final"],
        y_fav, dates, currencies, benefit,
    )
    _cross_period_robustness(
        "ordinary_states", ordinary_outputs, y_fav, dates, currencies, benefit
    )

    # A2. A separately labelled post-shock reset sensitivity. It cannot be
    # judged on the old general block because the training window starts in 2022.
    reset_eb_specs = [
        EBSpec("reset_eb_p20", None, 20.0, 0.0, 0.0),
        EBSpec("reset_eb_p50_lcb", None, 50.0, .5, 0.0),
        EBSpec("reset_eb_anchor25", None, 30.0, 0.0, .25),
        EBSpec("reset_eb_anchor50", None, 30.0, 0.0, .50),
    ]
    reset_markov_specs = [
        EBSpec("reset_markov_p20", None, 20.0, 0.0, 0.0),
        EBSpec("reset_markov_p50_lcb", None, 50.0, .5, 0.0),
        EBSpec("reset_markov_anchor25", None, 30.0, 0.0, .25),
        EBSpec("reset_markov_anchor50", None, 30.0, 0.0, .50),
    ]
    reset_outputs = _generate_postshock_lookup(
        reset_eb_specs, HierarchicalStates, X, names, y_fav, dates, currencies, reach
    )
    reset_outputs.update(_generate_postshock_lookup(
        reset_markov_specs, DirectionalMarkovStates,
        X, names, y_fav, dates, currencies, reach
    ))
    with (OUT / "postshock_states_outputs.pkl").open("wb") as fh:
        pickle.dump(reset_outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    reset_stage1, reset_confirm, reset_audit, reset_combined = _postshock_selection(
        reset_outputs, y_fav, dates, currencies, benefit
    )
    _bootstrap(
        "postshock_states", reset_outputs, reset_combined,
        y_fav, benefit, dates, currencies,
    )

    # B. Isolated after-publication information set.
    pub_matrix, pub_names, eligibility, next_rows = _publication_matrix(
        X, names, index, series, cols
    )
    pub_anchors = _publication_anchors(X, names, index, series, eligibility, next_rows)
    pub_specs = [
        ModelSpec(name, after_publication=True, anchor=name)
        for name in pub_anchors
    ] + [
        ModelSpec("pub_logit_expand", _logit, after_publication=True),
        ModelSpec("pub_hist_7y", _hist, 7, True),
        ModelSpec("pub_extra_7y", _extra, 7, True),
        ModelSpec("pub_xgb_7y", _xgb, 7, True),
    ]
    pub_outputs = _generate_models(
        pub_specs, pub_matrix, y_fav, dates, reach, ALL_YEARS,
        anchors=pub_anchors, eligibility=eligibility,
    )
    n_current = len(cols)
    ablations = (
        ("pub_logit_current_gate", pub_matrix[:, :n_current]),
        ("pub_logit_published_next", pub_matrix[:, n_current:2 * n_current]),
        ("pub_logit_known_summary", pub_matrix[:, 2 * n_current:]),
        ("pub_logit_next_and_known", pub_matrix[:, n_current:]),
    )
    for name, matrix in ablations:
        pub_outputs.update(_generate_models(
            [ModelSpec(name, _logit, after_publication=True)],
            matrix, y_fav, dates, reach, ALL_YEARS, eligibility=eligibility,
        ))
    pub_outputs["pub_blend_logit_extra"] = _combine_outputs(
        {name: pub_outputs[name] for name in ("pub_logit_expand", "pub_extra_7y")},
        currencies, (.5, .5),
    )
    pub_outputs["pub_blend_logit_margin"] = _combine_outputs(
        {name: pub_outputs[name] for name in ("pub_logit_expand", "known_margin")},
        currencies, (.7, .3),
    )
    pub_outputs["pub_consensus_models"] = _combine_outputs(
        {name: pub_outputs[name] for name in (
            "pub_logit_expand", "pub_extra_7y", "pub_hist_7y", "pub_xgb_7y"
        )}, currencies, (.30, .30, .20, .20),
    )
    pub_outputs["pub_consensus_with_margin"] = _combine_outputs(
        {name: pub_outputs[name] for name in (
            "pub_logit_expand", "pub_extra_7y", "pub_xgb_7y", "known_margin"
        )}, currencies, (.30, .25, .20, .25),
    )
    publication = _run_track(
        "after_publication", pub_outputs, y_fav, dates, currencies, benefit
    )
    _annual_selected(
        "after_publication", pub_outputs, publication["final"],
        y_fav, dates, currencies, benefit,
    )
    _cross_period_robustness(
        "after_publication", pub_outputs, y_fav, dates, currencies, benefit
    )
    _publication_feature_importance(
        pub_matrix, pub_names, y_fav, dates, reach, eligibility
    )

    # C. Secondary closing-window objective under the ordinary timestamp.
    y_close = targets["close_h5"]
    close_specs = [
        ModelSpec("close_upper_range", anchor="upper_range"),
        ModelSpec("close_trend_anchor", anchor="trend"),
        ModelSpec("close_logit_expand", _logit),
        ModelSpec("close_hist_7y", _hist, 7),
        ModelSpec("close_extra_7y", _extra, 7),
        ModelSpec("close_xgb_7y", _xgb, 7),
    ]
    close_anchors = {
        "upper_range": _anchor_score(X, names, "upper_range"),
        "trend": _anchor_score(X, names, "trend"),
    }
    close_outputs = _generate_models(
        close_specs, current, y_close, dates, reach, ALL_YEARS, anchors=close_anchors
    )
    # Reuse the state candidates without choosing them on the fav target.
    close_outputs.update(_generate_eb(
        _eb_specs(), X, names, y_close, dates, currencies, reach, ALL_YEARS
    ))
    close_outputs.update(_generate_markov(
        markov_specs, X, names, y_close, dates, currencies, reach, ALL_YEARS
    ))
    closing = _run_track(
        "window_closing", close_outputs, y_close, dates, currencies, benefit
    )
    _annual_selected(
        "window_closing", close_outputs, closing["final"],
        y_close, dates, currencies, benefit,
    )
    _cross_period_robustness(
        "window_closing", close_outputs, y_close, dates, currencies, benefit
    )

    protocol = {
        "general_years": GENERAL, "shock_years": SHOCK, "final_years": FINAL,
        "final_status": "retrospective; previously inspected",
        "ordinary_information": "features through current effective CBR rate only",
        "publication_information": "next effective rate and cross-section, usable only after publication",
        "target_main": "fav_h5: current <= min(next five publications)",
        "target_secondary": "close_h5: rate at publication i+5 > current",
        "frequency_requirement": "1-2 alerts per currency per week",
        "ordinary_candidates": sorted(ordinary_outputs),
        "postshock_reset_candidates": sorted(reset_outputs),
        "publication_candidates": sorted(pub_outputs),
        "closing_candidates": sorted(close_outputs),
        "publication_features": pub_names,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_leakage_audit(index, series, reach, targets, eligibility, next_rows)

    summary = []
    for track, result in (
        ("ordinary_states", ordinary),
        ("after_publication", publication),
        ("window_closing", closing),
    ):
        row = result["final"].iloc[0].to_dict()
        row["track"] = track
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "headline_summary.csv", index=False)


if __name__ == "__main__":
    main()
