"""Quarterly models with causal resolved-outcome state features."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.model_study import combine_outputs
from research.round2_statistical_audit import _bootstrap_all, _circular_shift_audit, _fired
from research.round5_adaptation import RESET, _anchor_outputs, _outputs, _quarter_starts, _next_quarter
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns
from research.round5_refit_calibration import _evaluate as rolling_metrics
from research.round6_rate_control import (
    Policy as WeeklyPolicy,
    _metrics as weekly_metrics,
    controlled_fired,
    policies as weekly_policies,
)
from research.round6_resolved_features import load_resolved_features


OUT = Path("results/research/round6/resolved_models")
OLD_OUTPUTS = Path("results/research/round5/adaptation/outputs.pkl")
SEED = 20260905
RATES = (.15, .18, .20, .22, .25, .30, .35)
ROLLING_POLICIES = ((20, 0), (40, 0), (60, 0), (120, 0),
                    (250, 0), (40, 3), (60, 3), (120, 3))


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    matrix: str


def specs():
    return [
        Spec("resolved_hist_compact", "hist", "compact"),
        Spec("resolved_hist_full", "hist", "full"),
        Spec("resolved_hist_deep", "hist_deep", "full"),
        Spec("resolved_extra_full", "extra", "full"),
        Spec("resolved_xgb_full", "xgb", "full"),
        Spec("resolved_logit_compact", "logit", "compact"),
        Spec("resolved_cat_full", "cat", "full"),
    ]


def _model(kind):
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=240, learning_rate=.035, max_leaf_nodes=9,
            min_samples_leaf=42, l2_regularization=15.0, random_state=SEED,
        )
    if kind == "hist_deep":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.025, max_leaf_nodes=17,
            min_samples_leaf=28, l2_regularization=24.0, random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            max_features=.65, n_jobs=-1, random_state=SEED,
        )
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=.025,
            min_child_weight=18, subsample=.80, colsample_bytree=.70,
            reg_lambda=20.0, reg_alpha=1.0, objective="binary:logistic",
            eval_metric="logloss", n_jobs=4, random_state=SEED,
        )
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.025, max_iter=3000, random_state=SEED),
        )
    if kind == "cat":
        return CatBoostClassifier(
            iterations=400, depth=5, learning_rate=.03, l2_leaf_reg=20.0,
            loss_function="Logloss", verbose=False, random_seed=SEED,
            thread_count=4, allow_writing_files=False,
        )
    raise KeyError(kind)


def prequential_scores(spec, matrix, y, dates, reach):
    scores = np.full(len(y), np.nan)
    log = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        test = (dates >= start) & (dates < end) & ~np.isnan(y)
        train = (
            np.asarray([value < start for value in reach])
            & (dates >= RESET) & ~np.isnan(y)
        )
        rows = np.where(train)[0]
        if not test.any() or len(rows) < 700:
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError("unresolved h=5 label admitted to resolved-state model")
        model = _model(spec.kind)
        model.fit(matrix[rows], y[rows])
        target = np.where(test)[0]
        scores[target] = model.predict_proba(matrix[target])[:, 1]
        log.append({
            "candidate": spec.name, "quarter": str(start),
            "n_train": len(rows), "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1],
        })
        print(f"  {spec.name:<27} quarter={start} train={len(rows):5d} "
              f"features={matrix.shape[1]:3d}", flush=True)
    return scores, log


def _policy_rows():
    rows = []
    for rate in RATES:
        for rolling, cooldown in ROLLING_POLICIES:
            rows.append({
                "policy_type": "rolling", "rate": rate, "rolling": rolling,
                "cooldown": cooldown, "history": 0, "strong": 0.0,
                "late": 0.0, "late_weekday": 0, "weekly_cap": 0,
            })
    for policy in weekly_policies():
        rows.append({
            "policy_type": "weekly", "rate": 0.0, "rolling": 0,
            "cooldown": 0, **asdict(policy),
        })
    return rows


def _weekly(row):
    return WeeklyPolicy(
        int(row["history"]), float(row["strong"]), float(row["late"]),
        int(row["late_weekday"]), int(row["weekly_cap"]),
    )


def _evaluate(output, years, row, y, benefit, dates, currencies):
    if row["policy_type"] == "rolling":
        return rolling_metrics(
            output, years, float(row["rate"]), int(row["rolling"]),
            int(row["cooldown"]), y, benefit, dates, currencies,
        )
    return weekly_metrics(
        output, years, _weekly(row), y, benefit, dates, currencies,
    )


def _fire(output, years, row, y, dates, currencies):
    if row["policy_type"] == "rolling":
        valid, fired = _fired(
            output, years, dates, currencies, y, float(row["rate"]),
            int(row["rolling"]), int(row["cooldown"]),
        )
        return valid, fired
    valid, fired, _score = controlled_fired(
        output, years, dates, currencies, y, _weekly(row),
    )
    return valid, fired


def _choose(part):
    feasible = part[
        part.frequency.between(1.00, 2.00)
        & part.corridor_freq_min.ge(.80)
        & part.quarter_frequency_min.ge(.70)
        & part.forward_benefit_bps.gt(0)
    ].copy()
    pool = feasible if len(feasible) else part.copy()
    pool["robustness"] = pool[["lift", "corridor_lift_min"]].min(axis=1)
    return pool.sort_values(
        ["robustness", "lift", "quarter_frequency_min"], ascending=False,
    ).iloc[0]


def _row_policy(row):
    return {
        "policy_type": str(row.policy_type), "rate": float(row.rate),
        "rolling": int(row.rolling), "cooldown": int(row.cooldown),
        "history": int(row.history), "strong": float(row.strong),
        "late": float(row.late), "late_weekday": int(row.late_weekday),
        "weekly_cap": int(row.weekly_cap),
    }


def _bootstrap(selected, outputs, years, y, benefit, dates, currencies):
    common = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
    masks = {}
    for row in selected.itertuples(index=False):
        valid, fired = _fire(
            outputs[row.candidate], years, _row_policy(row), y, dates, currencies,
        )
        if np.array_equal(valid, common):
            masks[row.candidate] = fired
    draws = _bootstrap_all(y, benefit, dates, common, masks)
    rows = []
    for name, fired in masks.items():
        active = common & fired
        lift = draws[name]["lift"]
        gain = draws[name]["benefit"]
        lift = lift[np.isfinite(lift)]
        gain = gain[np.isfinite(gain)]
        rows.append({
            "candidate": name, "n": int(active.sum()),
            "lift": float(y[active].mean() / y[common].mean()),
            "lift_ci_low": float(np.quantile(lift, .025)),
            "lift_ci_high": float(np.quantile(lift, .975)),
            "p_lift_le_1": float((np.sum(lift <= 1) + 1) / (len(lift) + 1)),
            "benefit_ci_low": float(np.quantile(gain, .025)),
            "benefit_ci_high": float(np.quantile(gain, .975)),
        })
    return pd.DataFrame(rows), masks, common


def _breakdown(candidate, output, years, policy, y, benefit, dates, currencies):
    valid, fired = _fire(output, years, policy, y, dates, currencies)
    groups = [("overall", "all", valid, len(CORRIDORS))]
    for year in years:
        groups.append((
            "year", str(year), valid & np.asarray([day.year == year for day in dates]),
            len(CORRIDORS),
        ))
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                day.year == year and (day.month - 1) // 3 + 1 == quarter
                for day in dates
            ])
            if scope.any():
                groups.append(("quarter", f"{year}Q{quarter}", scope, len(CORRIDORS)))
    for currency in CORRIDORS:
        groups.append(("currency", currency, valid & (currencies == currency), 1))
    rows = []
    for kind, group, scope, n_corridors in groups:
        active = scope & fired
        values = benefit[active & ~np.isnan(benefit)]
        base = float(y[scope].mean())
        hit = float(y[active].mean()) if active.any() else np.nan
        rows.append({
            "candidate": candidate, "breakdown": kind, "group": group,
            "n_scope": int(scope.sum()), "n_signals": int(active.sum()),
            "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
            "base_rate": base, "hit_rate": hit,
            "lift": hit / base if active.any() else np.nan,
            "forward_benefit_bps": float(np.mean(values)) if len(values) else np.nan,
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, trajectory, trajectory_names, _paths = load_round5_features()
    _X2, _names2, index2, _series2, resolved, resolved_names = load_resolved_features()
    if index != index2:
        raise AssertionError("resolved feature index mismatch")
    core = X[:, _core_columns(names)]
    summary = np.asarray([
        i for i, name in enumerate(trajectory_names) if not name.startswith("rocket_")
    ], dtype=int)
    matrices = {
        "compact": np.column_stack([core, resolved]),
        "full": np.column_stack([core, trajectory[:, summary], resolved]),
    }
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])

    outputs, training_log = {}, []
    for spec in specs():
        scores, rows = prequential_scores(
            spec, matrices[spec.matrix], y, dates, reach,
        )
        outputs[spec.name] = _outputs(scores, y, dates)
        training_log.extend(rows)
    anchor = _anchor_outputs(X, names, y, dates)
    outputs["anchor_multiscale_locked"] = anchor
    with OLD_OUTPUTS.open("rb") as handle:
        old = pickle.load(handle)
    outputs["round5_reset_hist"] = old["quarterly_reset_hist"]
    ensembles = {
        "resolved_hist_full_anchor25": (
            ("resolved_hist_full", "anchor_multiscale_locked"), (.75, .25)),
        "resolved_hist_full_anchor50": (
            ("resolved_hist_full", "anchor_multiscale_locked"), (.50, .50)),
        "resolved_hist_compact_anchor25": (
            ("resolved_hist_compact", "anchor_multiscale_locked"), (.75, .25)),
        "resolved_hist_compact_anchor50": (
            ("resolved_hist_compact", "anchor_multiscale_locked"), (.50, .50)),
        "resolved_tree_consensus": (
            ("resolved_hist_full", "resolved_extra_full"), (.50, .50)),
        "resolved_boost_consensus": (
            ("resolved_hist_full", "resolved_xgb_full", "resolved_cat_full"),
            (1/3, 1/3, 1/3)),
        "resolved_diverse_anchor": (
            ("resolved_hist_full", "resolved_extra_full", "resolved_xgb_full",
             "anchor_multiscale_locked"), (.25, .25, .25, .25)),
        "resolved_hist_old_hist": (
            ("resolved_hist_full", "round5_reset_hist"), (.50, .50)),
        "resolved_hist_old_anchor": (
            ("resolved_hist_full", "round5_reset_hist",
             "anchor_multiscale_locked"), (.40, .30, .30)),
    }
    for name, (members, weights) in ensembles.items():
        outputs[name] = combine_outputs(
            [outputs[member] for member in members], weights, currencies,
        )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

    policy_rows = _policy_rows()
    grid_rows = []
    for candidate, output in outputs.items():
        for policy in policy_rows:
            item = _evaluate(output, (2024,), policy, y, benefit, dates, currencies)
            item.update({"candidate": candidate, **policy})
            grid_rows.append(item)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "screen_2024_grid.csv", index=False)
    selected = pd.DataFrame([_choose(part) for _, part in grid.groupby("candidate")])
    selected = selected.sort_values(["robustness", "lift"], ascending=False)
    selected.to_csv(OUT / "screen_2024_selected.csv", index=False)

    confirmation, auditing, combined = [], [], []
    for row in selected.itertuples(index=False):
        policy = _row_policy(row)
        for years, target in (((2025,), confirmation), ((2026,), auditing),
                              ((2025, 2026), combined)):
            item = _evaluate(
                outputs[row.candidate], years, policy, y, benefit, dates, currencies,
            )
            item.update({"candidate": row.candidate, **policy})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            if years == (2025,):
                item["clears_1p30_gate"] = bool(
                    item["lift"] >= 1.30 and 1.00 <= item["frequency"] <= 2.00
                    and item["corridor_freq_min"] >= .80
                    and item["quarter_frequency_min"] >= .70
                    and item["forward_benefit_bps"] > 0
                )
            target.append(item)
    confirm = pd.DataFrame(confirmation).sort_values(
        ["clears_1p30_gate", "robustness", "lift"], ascending=False,
    )
    passed = set(confirm.loc[confirm.clears_1p30_gate, "candidate"])
    audit = pd.DataFrame(auditing)
    audit["passed_2025"] = audit.candidate.isin(passed)
    audit = audit.sort_values(["passed_2025", "robustness", "lift"], ascending=False)
    together = pd.DataFrame(combined)
    together["passed_2025"] = together.candidate.isin(passed)
    together = together.sort_values(
        ["passed_2025", "macro_year_lift", "lift"], ascending=False,
    )
    confirm.to_csv(OUT / "confirm_2025.csv", index=False)
    audit.to_csv(OUT / "audit_2026.csv", index=False)
    together.to_csv(OUT / "combined_2025_2026.csv", index=False)

    finalists = selected.head(8)
    boot_2025, masks_2025, valid_2025 = _bootstrap(
        finalists, outputs, (2025,), y, benefit, dates, currencies,
    )
    boot_2025["period"] = "2025"
    boot_both, masks_both, valid_both = _bootstrap(
        finalists, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    boot_both["period"] = "2025_2026"
    pd.concat([boot_2025, boot_both], ignore_index=True).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    pd.concat([
        _circular_shift_audit(
            y, dates, currencies, valid_2025, masks_2025, "confirmation_2025",
        ),
        _circular_shift_audit(
            y, dates, currencies, valid_both, masks_both,
            "retrospective_2025_2026",
        ),
    ], ignore_index=True).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)

    breakdown = []
    for row in finalists.itertuples(index=False):
        breakdown.extend(_breakdown(
            row.candidate, outputs[row.candidate], (2025, 2026), _row_policy(row),
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "finalist_breakdown.csv", index=False)

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("resolved-state training chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "next_rate_feature": False,
        "resolved_outcomes_require_reach_strictly_before_row": True,
        "policy_selected_on": 2024, "confirmation": 2025, "audit": 2026,
        "n_resolved_features": len(resolved_names),
        "n_architectures": len(outputs), "n_policies": len(policy_rows),
        "chronology_ok": chronology_ok, "pristine_holdout_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    show = ["candidate", "policy_type", "frequency", "lift",
            "forward_benefit_bps", "corridor_lift_min",
            "quarter_frequency_min", "clears_1p30_gate"]
    print("\n2024 SELECTED\n" + selected[[
        "candidate", "policy_type", "frequency", "lift", "corridor_lift_min",
        "quarter_frequency_min", "robustness",
    ]].head(15).to_string(index=False))
    print("\n2025 CONFIRMATION\n" + confirm[show].head(20).to_string(index=False))
    print("\n2026 AUDIT\n" + audit[[
        "candidate", "policy_type", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "passed_2025",
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
