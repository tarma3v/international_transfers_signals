"""Quarterly, leakage-safe adaptation study for the post-2022 regime."""
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
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.model_study import combine_outputs
from research.round2_statistical_audit import _bootstrap_all, _fired
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round5_novel_models import (
    POLICIES,
    RATES,
    _core_columns,
    _metrics,
)


OUT = Path("results/research/round5/adaptation")
SEED = 20260904
RESET = dt.date(2022, 2, 24)
YEARS = (2024, 2025, 2026)


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    reset: bool = False
    window_years: int | None = None
    half_life_years: float | None = None


def specs():
    return [
        Spec("quarterly_reset_hist", "hist", reset=True),
        Spec("quarterly_reset_extra", "extra", reset=True),
        Spec("quarterly_reset_logit", "logit", reset=True),
        Spec("quarterly_window2_hist", "hist", window_years=2),
        Spec("quarterly_window3_hist", "hist", window_years=3),
        Spec("quarterly_decay1_hist", "hist", half_life_years=1.0),
    ]


def _model(kind):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=220, learning_rate=.035, max_leaf_nodes=9,
            min_samples_leaf=42, l2_regularization=15.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=450, max_depth=7, min_samples_leaf=25,
            max_features=.70, n_jobs=-1, random_state=SEED,
        )
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.035, max_iter=2500, random_state=SEED),
        )
    raise KeyError(kind)


def _quarter_starts():
    return [dt.date(year, month, 1) for year in range(2023, 2027) for month in (1, 4, 7, 10)]


def _next_quarter(day):
    if day.month == 10:
        return dt.date(day.year + 1, 1, 1)
    return dt.date(day.year, day.month + 3, 1)


def _fit(model, X, y, train, weights):
    if weights is None:
        model.fit(X[train], y[train])
    elif hasattr(model, "named_steps"):
        model.fit(X[train], y[train], logisticregression__sample_weight=weights)
    else:
        model.fit(X[train], y[train], sample_weight=weights)
    return model


def prequential_scores(spec, X, y, dates, reach):
    scores = np.full(len(y), np.nan)
    train_sizes = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        test = (dates >= start) & (dates < end) & ~np.isnan(y)
        if not test.any():
            continue
        train = np.asarray([value < start for value in reach]) & ~np.isnan(y)
        if spec.reset:
            train &= dates >= RESET
        if spec.window_years:
            lower = dt.date(start.year - spec.window_years, start.month, 1)
            train &= dates >= lower
        rows = np.where(train)[0]
        if len(rows) < 700 or len(np.unique(y[rows])) < 2:
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError(f"unresolved h=5 label admitted at {start}")
        weights = None
        if spec.half_life_years:
            age = np.asarray([(start - dates[row]).days for row in rows], dtype=float)
            weights = np.power(.5, age / (365.25 * spec.half_life_years))
        model = _fit(_model(spec.kind), X, y, rows, weights)
        target = np.where(test)[0]
        scores[target] = model.predict_proba(X[target])[:, 1]
        train_sizes.append({"candidate": spec.name, "quarter": str(start),
                            "n_train": len(rows), "first_train": str(min(dates[rows])),
                            "last_resolved": str(max(reach[rows]))})
        print(f"  {spec.name:<28} quarter={start} train={len(rows):5d}", flush=True)
    return scores, train_sizes


def _outputs(scores, y, dates):
    result = {}
    for year in YEARS:
        calibration = (dates >= dt.date(year - 1, 1, 1)) & (dates < dt.date(year, 1, 1))
        test = np.asarray([day.year == year for day in dates])
        calibration &= ~np.isnan(y) & np.isfinite(scores)
        test &= ~np.isnan(y) & np.isfinite(scores)
        ca, te = np.where(calibration)[0], np.where(test)[0]
        result[year] = {"calib_idx": ca, "test_idx": te,
                        "calib_score": scores[ca], "test_score": scores[te]}
    return result


def _anchor_outputs(X, names, y, dates):
    score = (.5 * X[:, names.index("pct_range_90")]
             + .3 * X[:, names.index("pct_range_30")]
             + .2 * X[:, names.index("pct_range_180")])
    return _outputs(score, y, dates)


def _choose_2024(part):
    feasible = part[
        part.frequency.between(.90, 2.10)
        & part.corridor_freq_min.ge(.65)
        & part.forward_benefit_bps.gt(0)
    ].copy()
    pool = feasible if len(feasible) else part.copy()
    pool["robustness"] = pool[["lift", "corridor_lift_min"]].min(axis=1)
    return pool.sort_values(["robustness", "lift", "auc"], ascending=False).iloc[0]


def _grid(outputs, y, dates, currencies, benefit):
    rows = []
    for name, output in outputs.items():
        for rate in RATES:
            for rolling, cooldown in POLICIES:
                item = _metrics(output, (2024,), rate, rolling, cooldown,
                                y, dates, currencies, benefit)
                item.update({"candidate": name, "rate": rate,
                             "rolling": rolling or 0, "cooldown": cooldown})
                rows.append(item)
    return pd.DataFrame(rows)


def _bootstrap(finalists, outputs, anchor, y, benefit, dates, currencies):
    rows = []
    for period, years in (("2025", (2025,)), ("2026", (2026,)),
                          ("2025_2026", (2025, 2026))):
        valid = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
        policies = {}
        info = {"anchor_multiscale_locked": (anchor, .20, 250, 0)}
        for row in finalists.itertuples(index=False):
            info[row.candidate] = (outputs[row.candidate], float(row.stage1_rate),
                                   int(row.stage1_rolling) or None,
                                   int(row.stage1_cooldown))
        for name, (output, rate, rolling, cooldown) in info.items():
            actual_valid, fired = _fired(output, years, dates, currencies, y,
                                         rate, rolling, cooldown)
            if np.array_equal(valid, actual_valid):
                policies[name] = fired
        draws = _bootstrap_all(y, benefit, dates, valid, policies)
        anchor_active = valid & policies["anchor_multiscale_locked"]
        anchor_lift = float(y[anchor_active].mean() / y[valid].mean())
        for name, fired in policies.items():
            # Some moving-block samples contain no signals for a sparse policy.
            # Keep the paired bootstrap draw but exclude undefined ratios from
            # its quantiles and p-values instead of turning the full CI into NaN.
            lift_draws = draws[name]["lift"]
            benefit_draws = draws[name]["benefit"]
            anchor_draws = draws["anchor_multiscale_locked"]["lift"]
            finite_lift = lift_draws[np.isfinite(lift_draws)]
            finite_benefit = benefit_draws[np.isfinite(benefit_draws)]
            paired = np.isfinite(lift_draws) & np.isfinite(anchor_draws)
            difference = lift_draws[paired] - anchor_draws[paired]
            active = valid & fired
            active_benefit = benefit[active & ~np.isnan(benefit)]
            lift = float(y[active].mean() / y[valid].mean())
            item = {
                "policy": name, "n": int(active.sum()),
                "frequency": rate_per_week(int(active.sum()), len(CORRIDORS), dates, valid),
                "lift": lift,
                "forward_benefit_bps": float(np.mean(active_benefit)),
                "lift_ci_low": float(np.quantile(finite_lift, .025)),
                "lift_ci_high": float(np.quantile(finite_lift, .975)),
                "p_lift_le_1": float((np.sum(finite_lift <= 1.0) + 1)
                                     / (len(finite_lift) + 1)),
                "benefit_ci_low": float(np.quantile(finite_benefit, .025)),
                "benefit_ci_high": float(np.quantile(finite_benefit, .975)),
                "lift_diff_vs_anchor_ci_low": float(np.quantile(difference, .025)),
                "lift_diff_vs_anchor_ci_high": float(np.quantile(difference, .975)),
                "lift_diff_vs_anchor": float(lift - anchor_lift),
                "finite_bootstrap_lift": int(len(finite_lift)),
                "finite_bootstrap_difference": int(len(difference)),
                "period": period,
            }
            rows.append(item)
    return pd.DataFrame(rows)


def _winner_breakdown(outputs, winner, policy, y, benefit, dates, currencies):
    rate, rolling, cooldown = policy
    rows = []
    for period, years in (("2024", (2024,)), ("2025", (2025,)),
                          ("2026", (2026,)), ("2025_2026", (2025, 2026))):
        valid, fired = _fired(outputs[winner], years, dates, currencies, y,
                              rate, rolling, cooldown)
        groups = [("overall", "all", valid, len(CORRIDORS))]
        for year in years:
            scope = valid & np.asarray([day.year == year for day in dates])
            groups.append(("year", str(year), scope, len(CORRIDORS)))
        for currency in CORRIDORS:
            scope = valid & (currencies == currency)
            groups.append(("currency", currency, scope, 1))
        for year in years:
            for quarter in range(1, 5):
                scope = valid & np.asarray([
                    day.year == year and (day.month - 1) // 3 + 1 == quarter
                    for day in dates
                ])
                groups.append(("quarter", f"{year}Q{quarter}", scope, len(CORRIDORS)))
        for kind, group, scope, n_corridors in groups:
            active = scope & fired
            values = benefit[active & ~np.isnan(benefit)]
            base = float(y[scope].mean()) if scope.any() else np.nan
            hit = float(y[active].mean()) if active.any() else np.nan
            rows.append({
                "period": period, "candidate": winner,
                "breakdown": kind, "group": group,
                "n_scope": int(scope.sum()), "n_signals": int(active.sum()),
                "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
                "base_rate": base, "hit_rate": hit,
                "lift": hit / base if active.any() else np.nan,
                "forward_benefit_bps": float(np.mean(values)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def _winner_ablation(outputs, policy, y, benefit, dates, currencies):
    rate, rolling, cooldown = policy
    candidates = {
        "reset_hist_component_same_policy": "quarterly_reset_hist",
        "anchor_component_same_policy": "anchor_multiscale_locked",
        "rank_blend_50_50": "quarterly_reset_hist_anchor50",
    }
    rows = []
    for period, years in (("2024", (2024,)), ("2025", (2025,)),
                          ("2026", (2026,)), ("2025_2026", (2025, 2026))):
        for label, candidate in candidates.items():
            item = _metrics(outputs[candidate], years, rate, rolling, cooldown,
                            y, dates, currencies, benefit)
            item.update({"period": period, "ablation": label,
                         "rate": rate, "rolling": rolling or 0,
                         "cooldown": cooldown})
            rows.append(item)
        item = _metrics(outputs["anchor_multiscale_locked"], years, .20, 250, 0,
                        y, dates, currencies, benefit)
        item.update({"period": period, "ablation": "locked_anchor_reference",
                     "rate": .20, "rolling": 250, "cooldown": 0})
        rows.append(item)
    return pd.DataFrame(rows)


def _multiplicity_audit(stage1, outputs, y, dates, currencies):
    frames = []
    for period, years in (("confirmation_2025", (2025,)),
                          ("retrospective_2025_2026", (2025, 2026))):
        common_valid = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
        policies = {}
        for row in stage1.itertuples(index=False):
            valid, fired = _fired(
                outputs[row.candidate], years, dates, currencies, y,
                float(row.rate), int(row.rolling) or None, int(row.cooldown),
            )
            if np.array_equal(valid, common_valid) and fired.any():
                policies[row.candidate] = fired
        frames.append(_circular_shift_audit(
            y, dates, currencies, common_valid, policies, period,
        ))
    return pd.concat(frames, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, trajectory, trajectory_names, _paths = load_round5_features()
    summary_cols = np.asarray([i for i, name in enumerate(trajectory_names)
                               if not name.startswith("rocket_")], dtype=int)
    matrix = np.column_stack([X[:, _core_columns(names)], trajectory[:, summary_cols]])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    outputs = {}
    training_log = []
    for spec in specs():
        scores, rows = prequential_scores(spec, matrix, y, dates, reach)
        outputs[spec.name] = _outputs(scores, y, dates)
        training_log.extend(rows)
    anchor = _anchor_outputs(X, names, y, dates)
    outputs["anchor_multiscale_locked"] = anchor
    ensembles = {
        "quarterly_reset_tree_consensus": (
            ("quarterly_reset_hist", "quarterly_reset_extra"), (.5, .5)),
        "quarterly_reset_hist_anchor25": (
            ("quarterly_reset_hist", "anchor_multiscale_locked"), (.75, .25)),
        "quarterly_reset_hist_anchor50": (
            ("quarterly_reset_hist", "anchor_multiscale_locked"), (.5, .5)),
        "quarterly_decay_hist_anchor25": (
            ("quarterly_decay1_hist", "anchor_multiscale_locked"), (.75, .25)),
    }
    for name, (members, weights) in ensembles.items():
        outputs[name] = combine_outputs([outputs[member] for member in members], weights, currencies)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(training_log).to_csv(OUT / "training_log.csv", index=False)

    grid = _grid(outputs, y, dates, currencies, benefit)
    grid.to_csv(OUT / "screen_2024_grid.csv", index=False)
    stage1 = pd.DataFrame([_choose_2024(part) for _name, part in grid.groupby("candidate")])
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "screen_2024_selected.csv", index=False)

    gate_rows = []
    for row in stage1.itertuples(index=False):
        item = _metrics(outputs[row.candidate], (2025,), float(row.rate),
                        int(row.rolling) or None, int(row.cooldown),
                        y, dates, currencies, benefit)
        item.update({"candidate": row.candidate, "stage1_rate": row.rate,
                     "stage1_rolling": row.rolling,
                     "stage1_cooldown": row.cooldown})
        item["robustness"] = min(item["lift"], item["corridor_lift_min"])
        item["clears_1p30_gate"] = (
            item["lift"] >= 1.30 and item["frequency"] >= .90
            and item["frequency"] <= 2.10 and item["corridor_freq_min"] >= .65
            and item["forward_benefit_bps"] > 0
        )
        gate_rows.append(item)
    gate = pd.DataFrame(gate_rows).sort_values(["robustness", "lift"], ascending=False)
    gate.to_csv(OUT / "confirm_2025.csv", index=False)

    audit_rows = []
    for row in gate.itertuples(index=False):
        item = _metrics(outputs[row.candidate], (2026,), float(row.stage1_rate),
                        int(row.stage1_rolling) or None, int(row.stage1_cooldown),
                        y, dates, currencies, benefit)
        item.update({"candidate": row.candidate, "stage1_rate": row.stage1_rate,
                     "stage1_rolling": row.stage1_rolling,
                     "stage1_cooldown": row.stage1_cooldown,
                     "passed_2025": bool(row.clears_1p30_gate),
                     "status": "causal retrospective; 2026 seen before round 5"})
        item["robustness"] = min(item["lift"], item["corridor_lift_min"])
        audit_rows.append(item)
    audit = pd.DataFrame(audit_rows).sort_values(["passed_2025", "robustness", "lift"],
                                                  ascending=False)
    audit.to_csv(OUT / "audit_2026.csv", index=False)

    combined_rows = []
    for row in gate.itertuples(index=False):
        item = _metrics(outputs[row.candidate], (2025, 2026), float(row.stage1_rate),
                        int(row.stage1_rolling) or None, int(row.stage1_cooldown),
                        y, dates, currencies, benefit)
        item.update({"candidate": row.candidate,
                     "policy_selected_on": "2024 only",
                     "passed_2025": bool(row.clears_1p30_gate)})
        combined_rows.append(item)
    combined = pd.DataFrame(combined_rows).sort_values(
        ["passed_2025", "macro_year_lift", "lift"], ascending=False,
    )
    combined.to_csv(OUT / "combined_2025_2026.csv", index=False)

    finalists = gate.head(5)
    bootstrap = _bootstrap(finalists, outputs, anchor, y, benefit, dates, currencies)
    bootstrap.to_csv(OUT / "block_bootstrap.csv", index=False)

    passed = gate[gate.clears_1p30_gate]
    if len(passed):
        winner_row = passed.sort_values(["robustness", "lift"], ascending=False).iloc[0]
        winner = str(winner_row.candidate)
        winner_policy = (
            float(winner_row.stage1_rate),
            int(winner_row.stage1_rolling) or None,
            int(winner_row.stage1_cooldown),
        )
        _winner_breakdown(
            outputs, winner, winner_policy, y, benefit, dates, currencies,
        ).to_csv(OUT / "winner_breakdown.csv", index=False)
        _winner_ablation(
            outputs, winner_policy, y, benefit, dates, currencies,
        ).to_csv(OUT / "winner_ablation.csv", index=False)
    else:
        winner, winner_policy = None, None

    _multiplicity_audit(
        stage1, outputs, y, dates, currencies,
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)

    training_frame = pd.DataFrame(training_log)
    chronology_ok = bool(np.all(
        pd.to_datetime(training_frame.last_resolved)
        < pd.to_datetime(training_frame.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("training log contains unresolved future h=5 labels")
    (OUT / "leakage_audit.json").write_text(json.dumps({
        "next_rate_feature": False,
        "feature_information_cutoff": "row publication t",
        "target": "v[t] <= min(v[t+1:t+6])",
        "target_horizon_publications": 5,
        "quarterly_refit_uses_only_reach_date_before_refit": chronology_ok,
        "latest_train_reach_by_refit": training_frame.groupby("quarter")["last_resolved"].max().to_dict(),
        "winner": winner,
        "winner_policy": winner_policy,
        "pristine_holdout_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "protocol.json").write_text(json.dumps({
        "reset": str(RESET), "refit": "calendar quarter",
        "screen": 2024, "confirmation": 2025, "last_audit": 2026,
        "specs": [spec.__dict__ for spec in specs()],
        "ensembles": {name: {"members": members, "weights": weights}
                      for name, (members, weights) in ensembles.items()},
        "next_rate_feature": False,
        "winner": winner,
        "winner_policy": winner_policy,
        "pristine_holdout_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "corridor_freq_min", "corridor_lift_min", "robustness"]
    print("\n2024 SCREEN", stage1[columns].to_string(index=False), sep="\n")
    print("\n2025 CONFIRM", gate[columns + ["clears_1p30_gate"]].to_string(index=False), sep="\n")
    print("\n2026 AUDIT", audit[columns + ["passed_2025"]].to_string(index=False), sep="\n")
    print("\n2025-2026 COMBINED", combined[["candidate", "frequency", "lift",
          "macro_year_lift", "forward_benefit_bps", "year_lift_min",
          "corridor_lift_min", "passed_2025"]].to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
