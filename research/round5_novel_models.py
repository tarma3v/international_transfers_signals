"""Round-five ordinary-signal study with new, predeclared model families.

This module never exposes publication ``i+1`` to a feature or a score.  It
uses a preceding calibration year for policy thresholds and purges training
labels by their actual five-publication reach date.
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
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets, target_now_favourable
from ml.validation import target_reach_dates
from research.model_study import combine_outputs, evaluate
from research.round2_statistical_audit import (
    _bootstrap_all,
    _circular_shift_audit,
    _fired,
    _summary,
)
from research.round5_features import load_round5_features


OUT = Path("results/research/round5")
SEED = 20260904
GENERAL = (2017, 2018, 2019, 2020)
SHOCK = (2022, 2023)
FINAL = (2024, 2025, 2026)
ALL_YEARS = GENERAL + SHOCK + FINAL
RATES = (.18, .20, .22, .25, .30, .35, .40)
POLICIES = ((None, 0), (120, 0), (250, 0), (500, 0), (250, 3))


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    matrix: str
    factory: Callable[[], object]
    mode: str = "binary"
    stable_features: bool = False
    domain_balanced: bool = False


def _logit(C=.04):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2500, random_state=SEED),
    )


def _extra_classifier():
    return ExtraTreesClassifier(
        n_estimators=350, max_depth=8, min_samples_leaf=32,
        max_features=.65, n_jobs=-1, random_state=SEED,
    )


def _hist_classifier():
    return HistGradientBoostingClassifier(
        max_iter=220, learning_rate=.035, max_leaf_nodes=11,
        min_samples_leaf=60, l2_regularization=12.0, random_state=SEED,
    )


def _knn(neighbors):
    return make_pipeline(
        StandardScaler(), PCA(n_components=32, whiten=True, random_state=SEED),
        KNeighborsClassifier(n_neighbors=neighbors, weights="distance", p=2, n_jobs=-1),
    )


def _extra_regressor():
    return ExtraTreesRegressor(
        n_estimators=350, max_depth=8, min_samples_leaf=32,
        max_features=.65, n_jobs=-1, random_state=SEED,
    )


def _hist_regressor():
    return HistGradientBoostingRegressor(
        max_iter=220, learning_rate=.035, max_leaf_nodes=11,
        min_samples_leaf=60, l2_regularization=12.0, random_state=SEED,
    )


def specs() -> list[Spec]:
    return [
        Spec("invariant_summary_logit", "invariant", "summary", lambda: _logit(.04)),
        Spec("invariant_summary_extra", "invariant", "summary", _extra_classifier),
        Spec("invariant_summary_hist", "invariant", "summary", _hist_classifier),
        Spec("rocket_logit", "rocket", "rocket", lambda: _logit(.025)),
        Spec("stable_invariant_logit", "stable", "summary", lambda: _logit(.05),
             stable_features=True),
        Spec("domain_balanced_logit", "stable", "summary", lambda: _logit(.04),
             domain_balanced=True),
        Spec("path_analogue_knn100", "analogue", "paths", lambda: _knn(100)),
        Spec("path_analogue_knn250", "analogue", "paths", lambda: _knn(250)),
        Spec("ordinal_count_extra", "rich_target", "summary", _extra_regressor,
             mode="ordinal"),
        Spec("future_floor_extra", "rich_target", "summary", _extra_regressor,
             mode="floor"),
        Spec("future_floor_hist", "rich_target", "summary", _hist_regressor,
             mode="floor"),
    ]


def _core_columns(names: list[str]) -> np.ndarray:
    requested = [
        "pct_range_30", "pct_range_90", "pct_range_180",
        "range_pos_20", "range_pos_60", "range_pos_120", "range_pos_250",
        "rank_level_20", "rank_level_60", "rank_level_120", "rank_level_250",
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
        "raw_vol_5", "raw_vol_20", "raw_vol_60", "raw_vol_120",
        "vol_ratio_5_60", "vol_ratio_20_120", "positive_share_20",
        "positive_share_60", "ret_ac1_20", "ret_ac1_60",
        "streak_up", "streak_dn", "bars_since_min_30", "bars_since_max_30",
        "peer_ret_5_mean", "rel_to_peers_5", "peer_dispersion_5",
        "usd_raw_ret_5", "usd_raw_ret_20", "cny_raw_ret_5", "cny_raw_ret_20",
        "eur_raw_ret_5", "eur_raw_ret_20", "cnyusd_ret_20", "eurusd_ret_20",
        "annual_sin_1", "annual_cos_1", "annual_sin_2", "annual_cos_2",
        "dow_sin", "dow_cos", "gap_days",
    ]
    requested += [name for name in names if name.startswith("currency_")]
    missing = [name for name in requested if name not in names]
    if missing:
        raise KeyError(f"missing core features: {missing}")
    return np.asarray([names.index(name) for name in requested], dtype=int)


def _matrices(X, names, trajectory, trajectory_names, paths):
    core = X[:, _core_columns(names)].astype(np.float32)
    summary_cols = np.asarray([
        i for i, name in enumerate(trajectory_names) if not name.startswith("rocket_")
    ], dtype=int)
    return {
        "summary": np.column_stack([core, trajectory[:, summary_cols]]),
        "rocket": np.column_stack([core, trajectory]),
        "paths": np.column_stack([core, paths]),
    }


def _future_objects(series, index, causal_scale):
    y_steps = np.full((len(index), 5), np.nan)
    count = np.full(len(index), np.nan)
    floor = np.full(len(index), np.nan)
    benefit = np.full(len(index), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        values = series[currency].values
        if position + 5 >= len(values):
            continue
        future = values[position + 1:position + 6]
        for step in range(1, 6):
            y_steps[row, step - 1] = target_now_favourable(values, position, step)
        count[row] = float(np.sum(future >= values[position]))
        minimum = float(np.min(np.log(future / values[position])) * 10000.0)
        floor[row] = minimum / max(float(causal_scale[row]), 1.0)
        benefit[row] = benefit_forward_only(values, position, 5)
    return y_steps, count, floor, benefit


def _masks(year, dates, reach, y):
    test_start = dt.date(year, 1, 1)
    calibration_start = dt.date(year - 1, 1, 1)
    train = np.asarray([r < calibration_start for r in reach]) & ~np.isnan(y)
    calibration = (dates >= calibration_start) & (dates < test_start) & ~np.isnan(y)
    test = np.asarray([day.year == year for day in dates]) & ~np.isnan(y)
    return np.where(train)[0], np.where(calibration)[0], np.where(test)[0]


def _stable_columns(matrix, y, dates, train, keep=90):
    correlations = []
    for year in sorted({day.year for day in dates[train]}):
        rows = train[np.asarray([dates[row].year == year for row in train])]
        if len(rows) < 100 or len(np.unique(y[rows])) < 2:
            continue
        xx = matrix[rows].astype(float)
        yy = y[rows].astype(float)
        xx -= xx.mean(axis=0)
        yy -= yy.mean()
        denom = np.sqrt(np.sum(xx * xx, axis=0) * np.sum(yy * yy))
        corr = np.divide(xx.T @ yy, denom, out=np.zeros(matrix.shape[1]), where=denom > 1e-12)
        correlations.append(corr)
    corr = np.asarray(correlations)
    if not len(corr):
        return np.arange(min(keep, matrix.shape[1]))
    sign_agreement = np.abs(np.mean(np.sign(corr), axis=0))
    strength = np.median(np.abs(corr), axis=0) * sign_agreement
    return np.argsort(strength)[::-1][:keep]


def _domain_weights(dates, currencies, train):
    keys = [(dates[row].year, currencies[row]) for row in train]
    counts = pd.Series(keys).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=float)
    return weights / weights.mean()


def _fit(model, matrix, target, train, sample_weight=None):
    if sample_weight is None:
        model.fit(matrix[train], target[train])
    elif hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        model.fit(matrix[train], target[train], logisticregression__sample_weight=sample_weight)
    else:
        model.fit(matrix[train], target[train], sample_weight=sample_weight)
    return model


def generate_plain(spec, matrix, y, ordinal, floor, dates, currencies, reach):
    target = {"binary": y, "ordinal": ordinal, "floor": floor}[spec.mode]
    outputs = {}
    for year in ALL_YEARS:
        tr, ca, te = _masks(year, dates, reach, y)
        cols = _stable_columns(matrix, y, dates, tr) if spec.stable_features else np.arange(matrix.shape[1])
        weights = _domain_weights(dates, currencies, tr) if spec.domain_balanced else None
        model = _fit(spec.factory(), matrix[:, cols], target, tr, weights)
        if spec.mode == "binary":
            cs = model.predict_proba(matrix[ca][:, cols])[:, 1]
            ts = model.predict_proba(matrix[te][:, cols])[:, 1]
        else:
            cs = model.predict(matrix[ca][:, cols])
            ts = model.predict(matrix[te][:, cols])
        outputs[year] = {"calib_idx": ca, "test_idx": te,
                         "calib_score": cs, "test_score": ts}
        print(f"  {spec.name:<30} year={year} train={len(tr):5d} cols={len(cols):3d}", flush=True)
    return outputs


def generate_multihorizon(matrix, y_steps, y, dates, reach):
    result = {"multi_extra_geometric": {}, "multi_extra_lower": {}}
    for year in ALL_YEARS:
        tr, ca, te = _masks(year, dates, reach, y)
        model = _extra_classifier().fit(matrix[tr], y_steps[tr])
        ca_parts = np.column_stack([part[:, 1] for part in model.predict_proba(matrix[ca])])
        te_parts = np.column_stack([part[:, 1] for part in model.predict_proba(matrix[te])])
        ca_geometric = np.exp(np.mean(np.log(np.clip(ca_parts, 1e-6, 1.0)), axis=1))
        te_geometric = np.exp(np.mean(np.log(np.clip(te_parts, 1e-6, 1.0)), axis=1))
        for name, cs, ts in (
            ("multi_extra_geometric", ca_geometric, te_geometric),
            ("multi_extra_lower", np.quantile(ca_parts, .25, axis=1),
             np.quantile(te_parts, .25, axis=1)),
        ):
            result[name][year] = {"calib_idx": ca, "test_idx": te,
                                  "calib_score": cs, "test_score": ts}
        print(f"  {'multi_extra':<30} year={year} train={len(tr):5d}", flush=True)
    return result


def _rank(reference, values):
    ordered = np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def generate_era_experts(matrix, y, dates, currencies, reach, kind):
    names = (f"era_{kind}_median", f"era_{kind}_lower")
    outputs = {name: {} for name in names}
    for year in ALL_YEARS:
        tr, ca, te = _masks(year, dates, reach, y)
        start, stop = min(dates[tr]).year, year - 2
        eras = []
        left = start
        while left <= stop:
            right = min(left + 2, stop)
            rows = tr[np.asarray([left <= dates[row].year <= right for row in tr])]
            if len(rows) >= 700 and len(np.unique(y[rows])) == 2:
                eras.append(rows)
            left += 3
        cal_experts, test_experts = [], []
        for number, rows in enumerate(eras):
            model = _logit(.05) if kind == "logit" else ExtraTreesClassifier(
                n_estimators=220, max_depth=7, min_samples_leaf=25,
                max_features=.70, n_jobs=-1, random_state=SEED + number,
            )
            model.fit(matrix[rows], y[rows])
            raw_cal = model.predict_proba(matrix[ca])[:, 1]
            raw_test = model.predict_proba(matrix[te])[:, 1]
            ranked_cal = np.zeros(len(ca)); ranked_test = np.zeros(len(te))
            for currency in CORRIDORS:
                cm = currencies[ca] == currency
                tm = currencies[te] == currency
                ranked_cal[cm] = _rank(raw_cal[cm], raw_cal[cm])
                ranked_test[tm] = _rank(raw_cal[cm], raw_test[tm])
            cal_experts.append(ranked_cal); test_experts.append(ranked_test)
        cal_experts = np.asarray(cal_experts); test_experts = np.asarray(test_experts)
        if not len(cal_experts):
            raise RuntimeError(f"no completed era expert for {year}")
        for name, cs, ts in (
            (names[0], np.median(cal_experts, axis=0), np.median(test_experts, axis=0)),
            (names[1], np.quantile(cal_experts, .25, axis=0),
             np.quantile(test_experts, .25, axis=0)),
        ):
            outputs[name][year] = {"calib_idx": ca, "test_idx": te,
                                   "calib_score": cs, "test_score": ts}
        print(f"  era_{kind:<25} year={year} experts={len(eras)}", flush=True)
    return outputs


def _anchor_outputs(X, names, dates, reach, y):
    score = (
        .5 * X[:, names.index("pct_range_90")]
        + .3 * X[:, names.index("pct_range_30")]
        + .2 * X[:, names.index("pct_range_180")]
    )
    output = {}
    for year in ALL_YEARS:
        _tr, ca, te = _masks(year, dates, reach, y)
        output[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": score[ca], "test_score": score[te]}
    return output


def _metrics(output, years, rate, rolling, cooldown, y, dates, currencies, benefit):
    row = evaluate(output, y, dates, currencies, benefit, years, rate, rolling, cooldown)
    annual = [evaluate(output, y, dates, currencies, benefit, (year,), rate, rolling, cooldown)
              for year in years]
    row["macro_year_lift"] = float(np.mean([item["lift"] for item in annual]))
    row["year_frequency_min"] = float(np.min([item["frequency"] for item in annual]))
    row["year_frequency_max"] = float(np.max([item["frequency"] for item in annual]))
    return row


def _grid(outputs, y, dates, currencies, benefit, years):
    rows = []
    for name, output in outputs.items():
        for rate in RATES:
            for rolling, cooldown in POLICIES:
                row = _metrics(output, years, rate, rolling, cooldown,
                               y, dates, currencies, benefit)
                row.update({"candidate": name, "rate": rate,
                            "rolling": rolling or 0, "cooldown": cooldown})
                rows.append(row)
    return pd.DataFrame(rows)


def _choose(part):
    feasible = part[
        part.frequency.between(.90, 2.10)
        & part.year_frequency_min.ge(.75)
        & part.year_frequency_max.le(2.25)
        & part.corridor_freq_min.ge(.65)
        & part.forward_benefit_bps.gt(0)
    ].copy()
    pool = feasible if len(feasible) else part.copy()
    pool["robustness"] = pool[
        ["lift", "macro_year_lift", "year_lift_min", "corridor_lift_min"]
    ].min(axis=1)
    return pool.sort_values(["robustness", "macro_year_lift", "lift"], ascending=False).iloc[0]


def _bootstrap(selected, outputs, anchor, y, benefit, dates, currencies):
    rows = []
    for period, years in (("shock_2022_2023", SHOCK), ("retrospective_2024_2026", FINAL)):
        common_valid = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
        policies = {}
        policy_info = {"anchor_multiscale_locked": (anchor, .20, 250, 0)}
        for row in selected.itertuples(index=False):
            policy_info[row.candidate] = (
                outputs[row.candidate], float(row.stage1_rate),
                int(row.stage1_rolling) or None, int(row.stage1_cooldown),
            )
        for name, (output, rate, rolling, cooldown) in policy_info.items():
            valid, fired = _fired(output, years, dates, currencies, y, rate, rolling, cooldown)
            if np.array_equal(valid, common_valid):
                policies[name] = fired
        draws = _bootstrap_all(y, benefit, dates, common_valid, policies)
        anchor_fire = policies["anchor_multiscale_locked"] & common_valid
        anchor_lift = float(y[anchor_fire].mean() / y[common_valid].mean())
        for name, fired in policies.items():
            item = _summary(name, y, benefit, dates, common_valid, fired,
                            draws[name], draws["anchor_multiscale_locked"], anchor_lift)
            item["period"] = period
            rows.append(item)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, trajectory, trajectory_names, paths = load_round5_features()
    matrices = _matrices(X, names, trajectory, trajectory_names, paths)
    dates = np.asarray([day for _currency, _position, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _position, _day in index], dtype=object)
    targets = build_targets(series, index)
    y = targets["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    causal_scale = X[:, names.index("raw_vol_20")]
    y_steps, ordinal, floor, benefit = _future_objects(series, index, causal_scale)

    outputs = {}
    for spec in specs():
        outputs[spec.name] = generate_plain(
            spec, matrices[spec.matrix], y, ordinal, floor, dates, currencies, reach,
        )
    outputs.update(generate_multihorizon(matrices["summary"], y_steps, y, dates, reach))
    outputs.update(generate_era_experts(matrices["summary"], y, dates, currencies, reach, "logit"))
    outputs.update(generate_era_experts(matrices["summary"], y, dates, currencies, reach, "extra"))

    anchor = _anchor_outputs(X, names, dates, reach, y)
    outputs["anchor_multiscale_locked"] = anchor
    fixed_ensembles = {
        "new_triad_equal": (
            ("rocket_logit", "multi_extra_geometric", "era_extra_median"),
            (1 / 3, 1 / 3, 1 / 3),
        ),
        "new_triad_anchor25": (
            ("rocket_logit", "multi_extra_geometric", "era_extra_median",
             "anchor_multiscale_locked"),
            (.25, .25, .25, .25),
        ),
        "invariant_extra_anchor50": (
            ("invariant_summary_extra", "anchor_multiscale_locked"), (.5, .5),
        ),
        "era_extra_anchor25": (
            ("era_extra_median", "anchor_multiscale_locked"), (.75, .25),
        ),
    }
    for name, (members, weights) in fixed_ensembles.items():
        outputs[name] = combine_outputs([outputs[member] for member in members], weights, currencies)

    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    families = {spec.name: spec.family for spec in specs()}
    families.update({"multi_extra_geometric": "multi_horizon",
                     "multi_extra_lower": "multi_horizon",
                     "era_logit_median": "era", "era_logit_lower": "era",
                     "era_extra_median": "era", "era_extra_lower": "era",
                     "anchor_multiscale_locked": "anchor"})
    families.update({name: "ensemble" for name in fixed_ensembles})

    general = _grid(outputs, y, dates, currencies, benefit, GENERAL)
    general["family"] = general.candidate.map(families)
    general.to_csv(OUT / "general_grid_2017_2020.csv", index=False)
    stage1 = pd.DataFrame([_choose(part) for _name, part in general.groupby("candidate")])
    stage1["family"] = stage1.candidate.map(families)
    stage1 = stage1.sort_values(["robustness", "macro_year_lift", "lift"], ascending=False)
    stage1.to_csv(OUT / "stage1_selected_policies.csv", index=False)

    advanced = stage1.groupby("family", sort=False, group_keys=False).head(2)
    shock_rows = []
    for row in advanced.itertuples(index=False):
        item = _metrics(outputs[row.candidate], SHOCK, float(row.rate),
                        int(row.rolling) or None, int(row.cooldown),
                        y, dates, currencies, benefit)
        item.update({"candidate": row.candidate, "family": row.family,
                     "stage1_rate": row.rate, "stage1_rolling": row.rolling,
                     "stage1_cooldown": row.cooldown})
        item["robustness"] = min(item[key] for key in
                                 ("lift", "macro_year_lift", "year_lift_min", "corridor_lift_min"))
        shock_rows.append(item)
    shock = pd.DataFrame(shock_rows).sort_values(
        ["robustness", "macro_year_lift", "lift"], ascending=False,
    )
    shock["clears_1p30_gate"] = (
        shock.lift.ge(1.30) & shock.macro_year_lift.ge(1.30)
        & shock.frequency.between(.90, 2.10)
        & shock.year_frequency_min.ge(.75) & shock.year_frequency_max.le(2.25)
        & shock.corridor_freq_min.ge(.65) & shock.forward_benefit_bps.gt(0)
    )
    shock.to_csv(OUT / "stage2_shock_2022_2023.csv", index=False)

    finalists = shock.head(5)
    final_rows = []
    for row in finalists.itertuples(index=False):
        item = _metrics(outputs[row.candidate], FINAL, float(row.stage1_rate),
                        int(row.stage1_rolling) or None, int(row.stage1_cooldown),
                        y, dates, currencies, benefit)
        item.update({"candidate": row.candidate, "family": row.family,
                     "stage1_rate": row.stage1_rate,
                     "stage1_rolling": row.stage1_rolling,
                     "stage1_cooldown": row.stage1_cooldown,
                     "status": "retrospective; 2024-2026 inspected before round 5"})
        item["robustness"] = min(item[key] for key in
                                 ("lift", "macro_year_lift", "year_lift_min", "corridor_lift_min"))
        final_rows.append(item)
    final = pd.DataFrame(final_rows).sort_values(
        ["robustness", "macro_year_lift", "lift"], ascending=False,
    )
    final.to_csv(OUT / "final_2024_2026_retrospective.csv", index=False)

    # Diagnostic only: apply every general-selected policy to final without
    # allowing the result to alter the locked finalist ranking above.
    sensitivity = []
    for row in stage1.itertuples(index=False):
        item = _metrics(outputs[row.candidate], FINAL, float(row.rate),
                        int(row.rolling) or None, int(row.cooldown),
                        y, dates, currencies, benefit)
        item.update({"candidate": row.candidate, "family": row.family,
                     "status": "post-hoc sensitivity; forbidden for selection"})
        sensitivity.append(item)
    pd.DataFrame(sensitivity).sort_values("lift", ascending=False).to_csv(
        OUT / "all_candidates_final_sensitivity.csv", index=False,
    )

    bootstrap_selection = shock.head(5)[[
        "candidate", "stage1_rate", "stage1_rolling", "stage1_cooldown"
    ]]
    bootstrap = _bootstrap(bootstrap_selection, outputs, anchor, y, benefit, dates, currencies)
    bootstrap.to_csv(OUT / "block_bootstrap.csv", index=False)

    circular_frames = []
    for period, years in (("shock_2022_2023", SHOCK), ("retrospective_2024_2026", FINAL)):
        valid = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
        policies = {}
        for row in advanced.itertuples(index=False):
            actual_valid, fired = _fired(
                outputs[row.candidate], years, dates, currencies, y,
                float(row.rate), int(row.rolling) or None, int(row.cooldown),
            )
            if np.array_equal(valid, actual_valid) and fired.any():
                policies[row.candidate] = fired
        circular_frames.append(_circular_shift_audit(
            y, dates, currencies, valid, policies, period,
        ))
    pd.concat(circular_frames, ignore_index=True).to_csv(
        OUT / "circular_shift_multiplicity.csv", index=False,
    )

    (OUT / "protocol.json").write_text(json.dumps({
        "information_set": "ordinary: publications through i only; i+1 forbidden",
        "architecture_selection": GENERAL,
        "adverse_regime_gate": SHOCK,
        "retrospective_only": FINAL,
        "candidate_families": {name: family for name, family in families.items()},
        "policies_per_architecture": len(RATES) * len(POLICIES),
        "selection_rule": "max min(lift, macro-year, min-year, min-currency) under frequency and benefit constraints",
        "pristine_holdout_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "family", "frequency", "lift", "macro_year_lift",
               "forward_benefit_bps", "year_lift_min", "corridor_lift_min", "robustness"]
    print("\nGENERAL", stage1[columns].head(20).to_string(index=False), sep="\n")
    print("\nSHOCK", shock[columns + ["clears_1p30_gate"]].to_string(index=False), sep="\n")
    print("\nRETROSPECTIVE FINAL", final[columns].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
