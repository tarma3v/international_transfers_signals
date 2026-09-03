"""Large leakage-controlled model and ensemble study.

Protocol fixed before inspecting the final slice:
* development / feature screening: through 2016;
* general validation: 2017--2020;
* post-shock adaptation validation: 2022--2023;
* final chronological evaluation: 2024--2026.

Each yearly forecast uses a core training set ending before the preceding
calendar year.  That preceding year is calibration-only: thresholds are set
there separately per corridor, then applied unchanged to the next year.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

from ml.data import CORRIDORS
from ml.evaluate import bootstrap_ci, rate_per_week
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build

warnings.filterwarnings("ignore", category=UserWarning)

DEV_END = dt.date(2016, 12, 31)
GENERAL_VALID_YEARS = (2017, 2018, 2019, 2020)
REGIME_VALID_YEARS = (2022, 2023)
FINAL_TEST_YEARS = (2024, 2025, 2026)
SHOCK_DATE = dt.date(2022, 2, 24)
TARGET_RATES = (0.20, 0.25, 0.30, 0.35, 0.40)
OUT = Path("results/research")
SEED = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[], object] | None = None
    feature_set: str = "all"
    local: bool = False
    window_years: int | None = None
    half_life_years: float | None = None
    anchor: str | None = None


def make_logit(C=0.1, penalty="l2"):
    solver = "saga" if penalty != "l2" else "lbfgs"
    kwargs = {"l1_ratio": 0.25} if penalty == "elasticnet" else {}
    return Pipeline([
        ("scale", RobustScaler()),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=3000,
                                    random_state=SEED, **kwargs)),
    ])


def make_hist():
    return Pipeline([("clf", HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=45,
        l2_regularization=3.0, random_state=SEED,
    ))])


def make_rf():
    return Pipeline([("clf", RandomForestClassifier(
        n_estimators=450, max_depth=7, min_samples_leaf=35, max_features=0.55,
        n_jobs=-1, random_state=SEED,
    ))])


def make_extra():
    return Pipeline([("clf", ExtraTreesClassifier(
        n_estimators=450, max_depth=8, min_samples_leaf=30, max_features=0.65,
        n_jobs=-1, random_state=SEED,
    ))])


def make_cat():
    return Pipeline([("clf", CatBoostClassifier(
        iterations=550, depth=5, learning_rate=0.035, l2_leaf_reg=8,
        random_seed=SEED, verbose=0, allow_writing_files=False,
    ))])


def make_xgb():
    return Pipeline([("clf", XGBClassifier(
        n_estimators=550, max_depth=4, learning_rate=0.035, min_child_weight=30,
        subsample=0.8, colsample_bytree=0.75, reg_lambda=5.0, reg_alpha=0.2,
        n_jobs=-1, random_state=SEED, eval_metric="logloss",
    ))])


def candidate_library() -> list[Candidate]:
    return [
        Candidate("anchor_pct90", anchor="pct90"),
        Candidate("anchor_multiscale", anchor="multiscale"),
        Candidate("anchor_trend", anchor="trend"),
        Candidate("anchor_season", anchor="season"),
        Candidate("logit_top80_expand", lambda: make_logit(.1), "top80"),
        Candidate("logit_top140_5y", lambda: make_logit(.1), "top140", window_years=5),
        Candidate("elastic_top140_5y", lambda: make_logit(.08, "elasticnet"), "top140", window_years=5),
        Candidate("hist_top80_expand", make_hist, "top80"),
        Candidate("hist_top140_5y", make_hist, "top140", window_years=5),
        Candidate("rf_top80_5y", make_rf, "top80", window_years=5),
        Candidate("extra_top80_5y", make_extra, "top80", window_years=5),
        Candidate("cat_top80_expand", make_cat, "top80"),
        Candidate("cat_top140_5y", make_cat, "top140", window_years=5),
        Candidate("cat_top140_decay2", make_cat, "top140", half_life_years=2.0),
        Candidate("xgb_top80_expand", make_xgb, "top80"),
        Candidate("xgb_top140_5y", make_xgb, "top140", window_years=5),
        Candidate("xgb_top140_decay2", make_xgb, "top140", half_life_years=2.0),
        Candidate("local_logit_top80_5y", lambda: make_logit(.1), "top80", local=True,
                  window_years=5),
        Candidate("local_cat_top80_5y", make_cat, "top80", local=True, window_years=5),
    ]


def train_feature_sets(X, y, dates, names) -> dict[str, np.ndarray]:
    dev = np.array([d <= DEV_END for d in dates]) & ~np.isnan(y)
    score = []
    for j in range(X.shape[1]):
        x = X[dev, j]
        if np.std(x) < 1e-12:
            value = 0.0
        else:
            value = abs(float(np.corrcoef(pd.Series(x).rank(), y[dev])[0, 1]))
        score.append(value if np.isfinite(value) else 0.0)
    order = np.argsort(score)[::-1]
    # Force identity and the key anchor into every selected subset.
    forced = [j for j, n in enumerate(names) if n.startswith("currency_")]
    forced += [names.index(n) for n in ("pct_range_90", "pct_range_30", "pct_range_180")]

    def selected(k):
        return np.asarray(list(dict.fromkeys(list(order[:k]) + forced)), dtype=int)

    return {"all": np.arange(X.shape[1]), "top80": selected(80), "top140": selected(140)}


def anchor_score(kind: str, X, names) -> np.ndarray:
    col = lambda n: X[:, names.index(n)]
    if kind == "pct90":
        return col("pct_range_90")
    if kind == "multiscale":
        return .5 * col("pct_range_90") + .3 * col("pct_range_30") + .2 * col("pct_range_180")
    if kind == "trend":
        return col("pct_range_90") + .035 * col("ret_20") + .015 * col("ret_60")
    if kind == "season":
        # Month effects are chosen from pre-2017 EDA only: May/Aug/Nov were the
        # recurring high-hit months across corridors. Test years never choose them.
        bonus = 7.0 * (col("is_month_05") + col("is_month_08") + col("is_month_11"))
        return col("pct_range_90") + bonus
    raise KeyError(kind)


def _fit_predict(model, X, y, train, calib, test, weights=None):
    fit_kwargs = {"clf__sample_weight": weights} if weights is not None else {}
    model.fit(X[train], y[train], **fit_kwargs)
    return model.predict_proba(X[calib])[:, 1], model.predict_proba(X[test])[:, 1]


def generate_outputs(candidate: Candidate, X, y, dates, currencies, reach, names,
                     feature_sets, years) -> dict[int, dict[str, np.ndarray]]:
    cols = feature_sets[candidate.feature_set]
    out = {}
    for year in years:
        test_start = dt.date(year, 1, 1)
        calibration_start = dt.date(year - 1, 1, 1)
        train = np.array([r < calibration_start for r in reach]) & ~np.isnan(y)
        if candidate.window_years is not None:
            lower = dt.date(year - 1 - candidate.window_years, 1, 1)
            train &= np.array([d >= lower for d in dates])
        calib = np.array([calibration_start <= d < test_start for d in dates]) & ~np.isnan(y)
        test = np.array([d.year == year for d in dates]) & ~np.isnan(y)
        tr, ca, te = np.where(train)[0], np.where(calib)[0], np.where(test)[0]
        if min(len(tr), len(ca), len(te)) == 0:
            continue

        if candidate.anchor:
            all_score = anchor_score(candidate.anchor, X, names)
            cal_score, test_score = all_score[ca], all_score[te]
        elif candidate.local:
            cal_score, test_score = np.full(len(ca), np.nan), np.full(len(te), np.nan)
            for currency in CORRIDORS:
                trc = tr[currencies[tr] == currency]
                cap = np.where(currencies[ca] == currency)[0]
                tep = np.where(currencies[te] == currency)[0]
                weights = None
                if candidate.half_life_years:
                    age = np.array([(calibration_start - dates[r]).days for r in trc])
                    weights = np.power(.5, age / (365.25 * candidate.half_life_years))
                cs, ts = _fit_predict(candidate.factory(), X[:, cols], y, trc, ca[cap], te[tep], weights)
                cal_score[cap], test_score[tep] = cs, ts
        else:
            weights = None
            if candidate.half_life_years:
                age = np.array([(calibration_start - dates[r]).days for r in tr])
                weights = np.power(.5, age / (365.25 * candidate.half_life_years))
            cal_score, test_score = _fit_predict(
                candidate.factory(), X[:, cols], y, tr, ca, te, weights
            )
        out[year] = {"calib_idx": ca, "test_idx": te,
                     "calib_score": cal_score, "test_score": test_score}
        print(f"  {candidate.name:<28} year={year} train={len(tr):5d}", flush=True)
    return out


def _rank_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def combine_outputs(parts: list[dict], weights: tuple[float, ...], currencies) -> dict:
    result = {}
    years = sorted(set.intersection(*(set(p) for p in parts)))
    for year in years:
        ca, te = parts[0][year]["calib_idx"], parts[0][year]["test_idx"]
        cal_comb = np.zeros(len(ca)); test_comb = np.zeros(len(te))
        for part, weight in zip(parts, weights):
            cal_raw, test_raw = part[year]["calib_score"], part[year]["test_score"]
            for currency in CORRIDORS:
                cm = currencies[ca] == currency
                tm = currencies[te] == currency
                cal_comb[cm] += weight * _rank_against(cal_raw[cm], cal_raw[cm])
                test_comb[tm] += weight * _rank_against(cal_raw[cm], test_raw[tm])
        result[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": cal_comb, "test_score": test_comb}
    return result


def evaluate(outputs, y, dates, currencies, benefit, years, target_rate,
             rolling_window: int | None = None, cooldown_days: int = 0):
    scope = np.zeros(len(y), dtype=bool)
    fired = np.zeros(len(y), dtype=bool)
    score = np.full(len(y), np.nan)
    for year in years:
        if year not in outputs:
            continue
        z = outputs[year]
        ca, te = z["calib_idx"], z["test_idx"]
        scope[te] = True
        score[te] = z["test_score"]
        for currency in CORRIDORS:
            cm = currencies[ca] == currency
            tm = currencies[te] == currency
            cal_order = np.argsort(dates[ca[cm]])
            test_order = np.argsort(dates[te[tm]])
            cal_scores = z["calib_score"][cm][cal_order]
            test_rows = te[tm][test_order]
            test_scores = score[test_rows]
            if rolling_window:
                joined = np.concatenate([cal_scores, test_scores])
                cutoffs = (
                    pd.Series(joined)
                    .rolling(rolling_window, min_periods=1)
                    .quantile(1.0 - target_rate)
                    .shift(1)
                    .to_numpy()[len(cal_scores):]
                )
            else:
                cutoffs = np.full(
                    len(test_rows), float(np.quantile(cal_scores, 1.0 - target_rate))
                )
            last_fire = None
            for row, cutoff in zip(test_rows, cutoffs):
                enough_gap = last_fire is None or (dates[row] - last_fire).days >= cooldown_days
                if score[row] >= cutoff and enough_gap:
                    fired[row] = True
                    last_fire = dates[row]
    valid = scope & ~np.isnan(y)
    active = fired & valid
    base = float(y[valid].mean())
    hit = float(y[active].mean()) if active.any() else np.nan
    lift = hit / base
    freq = rate_per_week(int(active.sum()), len(CORRIDORS), dates, valid)
    auc = float(roc_auc_score(y[valid], score[valid]))
    b = benefit[active & ~np.isnan(benefit)]
    benefit_mean = float(np.mean(b)) if len(b) else np.nan
    year_lifts = []
    for year in years:
        ym = valid & np.array([d.year == year for d in dates])
        yf = active & np.array([d.year == year for d in dates])
        if ym.any() and yf.sum() >= 10:
            year_lifts.append(float(y[yf].mean() / y[ym].mean()))
    corridor_freq = []
    corridor_lift = []
    for currency in CORRIDORS:
        cm = valid & (currencies == currency)
        cf = active & (currencies == currency)
        corridor_freq.append(rate_per_week(int(cf.sum()), 1, dates, cm))
        corridor_lift.append(float(y[cf].mean() / y[cm].mean()) if cf.any() else np.nan)
    clustered = []
    for currency in CORRIDORS:
        ds = sorted(dates[active & (currencies == currency)])
        clustered.extend([(b - a).days <= 7 for a, b in zip(ds[:-1], ds[1:])])
    return {
        "rate_target": target_rate, "n": int(active.sum()), "frequency": freq,
        "rolling_window": rolling_window or 0, "cooldown_days": cooldown_days,
        "hit_rate": hit, "base_rate": base, "lift": lift, "auc": auc,
        "forward_benefit_bps": benefit_mean,
        "year_lift_min": min(year_lifts) if year_lifts else np.nan,
        "year_lift_median": float(np.median(year_lifts)) if year_lifts else np.nan,
        "year_lift_positive": int(sum(v > 1 for v in year_lifts)),
        "years": len(year_lifts), "corridor_freq_min": float(np.nanmin(corridor_freq)),
        "corridor_freq_max": float(np.nanmax(corridor_freq)),
        "corridor_lift_min": float(np.nanmin(corridor_lift)),
        "cluster_share_7d": float(np.mean(clustered)) if clustered else np.nan,
    }


def best_row(frame: pd.DataFrame) -> pd.Series:
    feasible = frame[
        (frame.frequency >= .90) & (frame.frequency <= 2.10)
        & (frame.corridor_freq_min >= .65) & (frame.forward_benefit_bps > 0)
        & (frame.year_lift_min > 1.0) & (frame.corridor_lift_min > 1.0)
    ]
    pool = feasible if len(feasible) else frame
    robustness = np.minimum.reduce([
        pool["lift"].to_numpy(), pool["year_lift_min"].to_numpy(),
        pool["corridor_lift_min"].to_numpy(),
    ])
    pool = pool.assign(robustness=robustness)
    return pool.sort_values(["robustness", "lift", "auc"], ascending=False).iloc[0]


def simplex_weights(n: int, step: float = .2):
    units = int(round(1 / step))
    for bars in itertools.product(range(units + 1), repeat=n):
        if sum(bars) == units:
            yield tuple(v / units for v in bars)


def run_horizon(h: int = 5):
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series = load_or_build()
    dates = np.array([d for _c, _i, d in index], dtype=object)
    currencies = np.array([c for c, _i, _d in index], dtype=object)
    y = build_targets(series, index)[f"fav_h{h}"]
    reach = target_reach_dates(index, series, h)
    benefit = np.full(len(index), np.nan)
    for r, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, h)
        if value is not None:
            benefit[r] = value
    feature_sets = train_feature_sets(X, y, dates, names)
    pd.DataFrame({k: pd.Series([names[j] for j in v]) for k, v in feature_sets.items()}).to_csv(
        OUT / f"feature_sets_h{h}.csv", index=False
    )

    all_years = tuple(sorted(set(GENERAL_VALID_YEARS + REGIME_VALID_YEARS + FINAL_TEST_YEARS)))
    outputs = {}
    validation_rows = []
    candidates = candidate_library()
    prediction_cache = OUT / f"candidate_outputs_h{h}_v2.pkl"
    if prediction_cache.exists():
        with prediction_cache.open("rb") as fh:
            outputs = pickle.load(fh)
        print(f"Loaded cached candidate predictions: {prediction_cache}", flush=True)
    else:
        for candidate in candidates:
            outputs[candidate.name] = generate_outputs(
                candidate, X, y, dates, currencies, reach, names, feature_sets, all_years
            )
        with prediction_cache.open("wb") as fh:
            pickle.dump(outputs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    for candidate in candidates:
        for rate in TARGET_RATES:
            row = evaluate(outputs[candidate.name], y, dates, currencies, benefit,
                           GENERAL_VALID_YEARS, rate)
            row["candidate"] = candidate.name
            validation_rows.append(row)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / f"general_validation_h{h}.csv", index=False)

    # Family selection is based exclusively on 2017--2020.
    per_candidate = pd.DataFrame([best_row(z) for _n, z in validation.groupby("candidate")])
    top_names = list(per_candidate.sort_values("lift", ascending=False).candidate.head(5))
    # The structural-break hypothesis was declared before evaluation, so the
    # transparent price anchors remain eligible in the 2022--2023 regime even
    # when they were weak in the pre-shock years.
    top_names = list(dict.fromkeys(top_names + [
        "anchor_pct90", "anchor_multiscale", "anchor_trend", "anchor_season"
    ]))
    print("\nTop general-validation candidates:")
    print(per_candidate.sort_values("lift", ascending=False)[
        ["candidate", "rate_target", "frequency", "lift", "forward_benefit_bps",
         "year_lift_min", "corridor_lift_min"]
    ].head(10).to_string(index=False))

    # Tune blend and rate on post-shock adaptation years only. Include every
    # selected single model and convex three-model ensembles.
    regime_rows = []
    blend_specs = []
    for name in top_names:
        blend_specs.append((name, (name,), (1.0,)))
    anchor_names = [n for n in top_names if n.startswith("anchor_")]
    ml_names = [n for n in top_names if not n.startswith("anchor_")]
    # Targeted blends cover anchor/ML complementarity without an expensive,
    # statistically fragile combinatorial search.
    for anchor in anchor_names:
        for model in ml_names:
            for anchor_weight in (.25, .50, .75):
                members = (anchor, model)
                weights = (anchor_weight, 1.0 - anchor_weight)
                label = "blend:" + "+".join(
                    f"{w:.2f}*{n}" for w, n in zip(weights, members)
                )
                blend_specs.append((label, members, weights))
    for left, right in itertools.combinations(ml_names, 2):
        label = f"blend:0.50*{left}+0.50*{right}"
        blend_specs.append((label, (left, right), (.5, .5)))
    for anchor in anchor_names[:2]:
        for left, right in itertools.combinations(ml_names[:4], 2):
            members = (anchor, left, right)
            weights = (.5, .25, .25)
            label = "blend:" + "+".join(
                f"{w:.2f}*{n}" for w, n in zip(weights, members)
            )
            blend_specs.append((label, members, weights))
    policy_grid = ((None, 0), (120, 0), (250, 0), (500, 0), (250, 3), (250, 5))
    for label, members, weights in blend_specs:
        combined = combine_outputs([outputs[n] for n in members], weights, currencies)
        for rate in TARGET_RATES:
            for rolling_window, cooldown_days in policy_grid:
                row = evaluate(
                    combined, y, dates, currencies, benefit, REGIME_VALID_YEARS,
                    rate, rolling_window=rolling_window, cooldown_days=cooldown_days,
                )
                row["candidate"] = label
                row["members"] = json.dumps(members)
                row["weights"] = json.dumps(weights)
                regime_rows.append(row)
    regime = pd.DataFrame(regime_rows)
    regime.to_csv(OUT / f"postshock_validation_h{h}.csv", index=False)
    winner = best_row(regime)
    print("\nLocked on post-shock validation 2022--2023:")
    print(winner[["candidate", "rate_target", "rolling_window", "cooldown_days",
                  "frequency", "lift", "forward_benefit_bps", "year_lift_min",
                  "corridor_lift_min"]].to_string())

    members = tuple(json.loads(winner["members"]))
    weights = tuple(json.loads(winner["weights"]))
    locked = combine_outputs([outputs[n] for n in members], weights, currencies)
    final = evaluate(
        locked, y, dates, currencies, benefit, FINAL_TEST_YEARS,
        float(winner.rate_target), rolling_window=int(winner.rolling_window) or None,
        cooldown_days=int(winner.cooldown_days),
    )
    final.update({"candidate": winner.candidate, "members": json.dumps(members),
                  "weights": json.dumps(weights), "h": h,
                  "selection_general_years": str(GENERAL_VALID_YEARS),
                  "selection_postshock_years": str(REGIME_VALID_YEARS),
                  "final_years": str(FINAL_TEST_YEARS)})
    pd.DataFrame([final]).to_csv(OUT / f"locked_final_h{h}.csv", index=False)
    print("\nFINAL 2024--2026 (read once after lock):")
    print(pd.Series(final).to_string())
    return final, winner, per_candidate


if __name__ == "__main__":
    run_horizon(5)
