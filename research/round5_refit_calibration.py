"""Error-driven, leakage-safe calibration across quarterly model refits.

The model is fitted only on resolved h=5 labels.  At each refit its raw score
is converted to a per-currency percentile using past rows scored by that same
fitted model, so a rolling alert threshold never compares probabilities from
incompatible model versions.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.model_study import combine_outputs
from research.round2_statistical_audit import _circular_shift_audit, _fired
from research.round5_adaptation import (
    RESET,
    _anchor_outputs,
    _bootstrap,
    _fit,
    _model,
    _next_quarter,
    _outputs,
    _quarter_starts,
    _winner_breakdown,
)
from research.round5_features import load_round5_features
from research.round5_novel_models import _core_columns, _metrics


OUT = Path("results/research/round5/refit_calibration")
RATES = (.12, .15, .18, .20, .22, .25, .30, .35)
POLICIES = ((None, 0), (20, 0), (40, 0), (60, 0), (120, 0),
            (250, 0), (40, 3), (60, 3))


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    rank_window: int


def specs():
    return [
        Spec("quarterly_reset_hist_qrank60", "hist", 60),
        Spec("quarterly_reset_hist_qrank120", "hist", 120),
        Spec("quarterly_reset_hist_qrank250", "hist", 250),
        Spec("quarterly_reset_extra_qrank120", "extra", 120),
    ]


def _rank(reference, values):
    ordered = np.sort(np.asarray(reference)[np.isfinite(reference)])
    if not len(ordered):
        return np.full(len(values), np.nan)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def prequential_rank_scores(spec, X, y, dates, currencies, reach):
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
        if not test.any() or len(rows) < 700 or len(np.unique(y[rows])) < 2:
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError(f"unresolved h=5 label admitted at {start}")
        model = _fit(_model(spec.kind), X, y, rows, None)
        target = np.where(test)[0]
        for currency in CORRIDORS:
            reference = np.where((dates < start) & (currencies == currency))[0]
            reference = reference[np.argsort(dates[reference])][-spec.rank_window:]
            current = target[currencies[target] == currency]
            if not len(reference) or not len(current):
                continue
            if not all(dates[row] < start for row in reference):
                raise AssertionError("refit calibration used a non-past row")
            reference_score = model.predict_proba(X[reference])[:, 1]
            current_score = model.predict_proba(X[current])[:, 1]
            scores[current] = _rank(reference_score, current_score)
        log.append({
            "candidate": spec.name, "quarter": str(start),
            "n_train": len(rows), "first_train": str(min(dates[rows])),
            "last_resolved": str(max(reach[rows])),
            "rank_window": spec.rank_window,
        })
        print(f"  {spec.name:<34} quarter={start} train={len(rows):5d}", flush=True)
    return scores, log


def _cadence(output, years, rate, rolling, cooldown, y, dates, currencies):
    valid, fired = _fired(output, years, dates, currencies, y, rate, rolling, cooldown)
    values = []
    for year in years:
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                day.year == year and (day.month - 1) // 3 + 1 == quarter
                for day in dates
            ])
            if scope.any():
                values.append(rate_per_week(
                    int((scope & fired).sum()), len(CORRIDORS), dates, scope,
                ))
    return {
        "quarter_frequency_min": float(min(values)),
        "quarter_frequency_max": float(max(values)),
        "quarters": len(values),
    }


def _evaluate(output, years, rate, rolling, cooldown, y, benefit, dates, currencies):
    item = _metrics(output, years, rate, rolling, cooldown,
                    y, dates, currencies, benefit)
    item.update(_cadence(output, years, rate, rolling, cooldown,
                         y, dates, currencies))
    return item


def _choose(part):
    feasible = part[
        part.frequency.between(.90, 2.10)
        & part.corridor_freq_min.ge(.65)
        & part.quarter_frequency_min.ge(.30)
        & part.quarter_frequency_max.le(2.50)
        & part.forward_benefit_bps.gt(0)
    ].copy()
    pool = feasible if len(feasible) else part.copy()
    pool["robustness"] = pool[["lift", "corridor_lift_min"]].min(axis=1)
    return pool.sort_values(["robustness", "lift", "auc"], ascending=False).iloc[0]


def _multiplicity(stage1, outputs, y, dates, currencies):
    frames = []
    for period, years in (("confirmation_2025", (2025,)),
                          ("retrospective_2025_2026", (2025, 2026))):
        common = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
        policies = {}
        for row in stage1.itertuples(index=False):
            valid, fired = _fired(
                outputs[row.candidate], years, dates, currencies, y,
                float(row.rate), int(row.rolling) or None, int(row.cooldown),
            )
            if np.array_equal(valid, common) and fired.any():
                policies[row.candidate] = fired
        frames.append(_circular_shift_audit(
            y, dates, currencies, common, policies, period,
        ))
    return pd.concat(frames, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, trajectory, trajectory_names, _paths = load_round5_features()
    summary = np.asarray([
        i for i, name in enumerate(trajectory_names) if not name.startswith("rocket_")
    ], dtype=int)
    matrix = np.column_stack([X[:, _core_columns(names)], trajectory[:, summary]])
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
        score, rows = prequential_rank_scores(
            spec, matrix, y, dates, currencies, reach,
        )
        outputs[spec.name] = _outputs(score, y, dates)
        training_log.extend(rows)
    anchor = _anchor_outputs(X, names, y, dates)
    outputs["anchor_multiscale_locked"] = anchor

    ensembles = {}
    for window in (60, 120, 250):
        base = f"quarterly_reset_hist_qrank{window}"
        for anchor_weight in (.25, .50):
            name = f"{base}_anchor{int(anchor_weight * 100)}"
            ensembles[name] = ((base, "anchor_multiscale_locked"),
                               (1.0 - anchor_weight, anchor_weight))
    ensembles.update({
        "quarterly_qrank_tree_consensus": (
            ("quarterly_reset_hist_qrank120", "quarterly_reset_extra_qrank120"),
            (.5, .5),
        ),
        "quarterly_qrank_tree_anchor": (
            ("quarterly_reset_hist_qrank120", "quarterly_reset_extra_qrank120",
             "anchor_multiscale_locked"),
            (.375, .375, .25),
        ),
    })
    for name, (members, weights) in ensembles.items():
        outputs[name] = combine_outputs(
            [outputs[member] for member in members], weights, currencies,
        )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(training_log)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate, output in outputs.items():
        for rate in RATES:
            for rolling, cooldown in POLICIES:
                item = _evaluate(output, (2024,), rate, rolling, cooldown,
                                 y, benefit, dates, currencies)
                item.update({"candidate": candidate, "rate": rate,
                             "rolling": rolling or 0, "cooldown": cooldown})
                rows.append(item)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "screen_2024_grid.csv", index=False)
    stage1 = pd.DataFrame([_choose(part) for _, part in grid.groupby("candidate")])
    stage1 = stage1.sort_values(["robustness", "lift"], ascending=False)
    stage1.to_csv(OUT / "screen_2024_selected.csv", index=False)

    gate_rows = []
    for row in stage1.itertuples(index=False):
        item = _evaluate(
            outputs[row.candidate], (2025,), float(row.rate),
            int(row.rolling) or None, int(row.cooldown),
            y, benefit, dates, currencies,
        )
        item.update({
            "candidate": row.candidate, "stage1_rate": row.rate,
            "stage1_rolling": row.rolling, "stage1_cooldown": row.cooldown,
        })
        item["robustness"] = min(item["lift"], item["corridor_lift_min"])
        item["clears_1p30_gate"] = bool(
            item["lift"] >= 1.30
            and .90 <= item["frequency"] <= 2.10
            and item["corridor_freq_min"] >= .65
            and item["quarter_frequency_min"] >= .30
            and item["quarter_frequency_max"] <= 2.50
            and item["forward_benefit_bps"] > 0
        )
        gate_rows.append(item)
    gate = pd.DataFrame(gate_rows).sort_values(
        ["clears_1p30_gate", "robustness", "lift"], ascending=False,
    )
    gate.to_csv(OUT / "confirm_2025.csv", index=False)

    audit_rows, combined_rows = [], []
    for row in gate.itertuples(index=False):
        policy = (float(row.stage1_rate), int(row.stage1_rolling) or None,
                  int(row.stage1_cooldown))
        item = _evaluate(outputs[row.candidate], (2026,), *policy,
                         y, benefit, dates, currencies)
        item.update({
            "candidate": row.candidate, "passed_2025": bool(row.clears_1p30_gate),
            "stage1_rate": policy[0], "stage1_rolling": policy[1] or 0,
            "stage1_cooldown": policy[2],
        })
        item["robustness"] = min(item["lift"], item["corridor_lift_min"])
        audit_rows.append(item)
        both = _evaluate(outputs[row.candidate], (2025, 2026), *policy,
                         y, benefit, dates, currencies)
        both.update({"candidate": row.candidate,
                     "passed_2025": bool(row.clears_1p30_gate)})
        combined_rows.append(both)
    audit = pd.DataFrame(audit_rows).sort_values(
        ["passed_2025", "robustness", "lift"], ascending=False,
    )
    combined = pd.DataFrame(combined_rows).sort_values(
        ["passed_2025", "macro_year_lift", "lift"], ascending=False,
    )
    audit.to_csv(OUT / "audit_2026.csv", index=False)
    combined.to_csv(OUT / "combined_2025_2026.csv", index=False)

    _bootstrap(gate.head(5), outputs, anchor, y, benefit, dates, currencies).to_csv(
        OUT / "block_bootstrap.csv", index=False,
    )
    _multiplicity(stage1, outputs, y, dates, currencies).to_csv(
        OUT / "circular_shift_multiplicity.csv", index=False,
    )

    passed = gate[gate.clears_1p30_gate]
    winner = None
    if len(passed):
        chosen = passed.sort_values(["robustness", "lift"], ascending=False).iloc[0]
        winner = str(chosen.candidate)
        policy = (float(chosen.stage1_rate), int(chosen.stage1_rolling) or None,
                  int(chosen.stage1_cooldown))
        _winner_breakdown(
            outputs, winner, policy, y, benefit, dates, currencies,
        ).to_csv(OUT / "winner_breakdown.csv", index=False)

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    if not chronology_ok:
        raise AssertionError("unresolved target admitted to a refit")
    (OUT / "protocol.json").write_text(json.dumps({
        "next_rate_feature": False,
        "quarterly_refit": True,
        "same_model_past_score_rank_windows": [60, 120, 250],
        "score_rank_uses_targets": False,
        "screen": 2024, "confirmation": 2025, "audit": 2026,
        "quarter_frequency_gate": [.30, 2.50],
        "winner": winner,
        "chronology_ok": chronology_ok,
        "pristine_holdout_available": False,
        "n_candidate_architectures": len(outputs),
        "n_policies_per_architecture": len(RATES) * len(POLICIES),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    show = ["candidate", "frequency", "lift", "forward_benefit_bps",
            "corridor_lift_min", "quarter_frequency_min",
            "quarter_frequency_max", "clears_1p30_gate"]
    print("\n2025 CONFIRMATION\n" + gate[show].to_string(index=False))
    print("\n2026 AUDIT\n" + audit[[
        "candidate", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min", "quarter_frequency_max",
        "passed_2025",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
