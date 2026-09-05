"""Cross-fitted score calibration for quarterly post-2022 models."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.model_study import combine_outputs
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
from research.round5_novel_models import _core_columns
from research.round5_refit_calibration import (
    POLICIES,
    RATES,
    _choose,
    _evaluate,
    _multiplicity,
    _rank,
)


OUT = Path("results/research/round5/crossfit_calibration")


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    heldout_quarters: int


def specs():
    return [
        Spec("quarterly_crossfit_hist_q1", "hist", 1),
        Spec("quarterly_crossfit_hist_q2", "hist", 2),
        Spec("quarterly_crossfit_extra_q1", "extra", 1),
        Spec("quarterly_crossfit_extra_q2", "extra", 2),
    ]


def _previous_quarter(day, count=1):
    value = day
    for _ in range(count):
        if value.month == 1:
            value = dt.date(value.year - 1, 10, 1)
        else:
            value = dt.date(value.year, value.month - 3, 1)
    return value


def prequential_crossfit_scores(spec, X, y, dates, currencies, reach):
    scores = np.full(len(y), np.nan)
    log = []
    for start in _quarter_starts():
        end = _next_quarter(start)
        calibration_start = _previous_quarter(start, spec.heldout_quarters)
        test = (dates >= start) & (dates < end) & ~np.isnan(y)
        train = (
            np.asarray([value < calibration_start for value in reach])
            & (dates >= RESET) & ~np.isnan(y)
        )
        rows = np.where(train)[0]
        reference = np.where((dates >= calibration_start) & (dates < start))[0]
        if (not test.any() or len(rows) < 500 or len(reference) < 200
                or len(np.unique(y[rows])) < 2):
            continue
        if not all(reach[row] < calibration_start for row in rows):
            raise AssertionError("training label overlaps held-out calibration quarter")
        if not all(dates[row] < start for row in reference):
            raise AssertionError("cross-fit reference contains the current quarter")
        model = _fit(_model(spec.kind), X, y, rows, None)
        target = np.where(test)[0]
        for currency in CORRIDORS:
            ref = reference[currencies[reference] == currency]
            current = target[currencies[target] == currency]
            if not len(ref) or not len(current):
                continue
            ref_score = model.predict_proba(X[ref])[:, 1]
            current_score = model.predict_proba(X[current])[:, 1]
            scores[current] = _rank(ref_score, current_score)
        log.append({
            "candidate": spec.name, "quarter": str(start),
            "heldout_start": str(calibration_start),
            "heldout_quarters": spec.heldout_quarters,
            "n_train": len(rows), "n_calibration": len(reference),
            "last_resolved": str(max(reach[rows])),
        })
        print(f"  {spec.name:<31} quarter={start} train={len(rows):5d} "
              f"cal={len(reference):4d}", flush=True)
    return scores, log


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
        score, rows = prequential_crossfit_scores(
            spec, matrix, y, dates, currencies, reach,
        )
        outputs[spec.name] = _outputs(score, y, dates)
        training_log.extend(rows)
    anchor = _anchor_outputs(X, names, y, dates)
    outputs["anchor_multiscale_locked"] = anchor
    ensembles = {
        "quarterly_crossfit_tree_consensus": (
            ("quarterly_crossfit_hist_q1", "quarterly_crossfit_extra_q1"), (.5, .5)),
        "quarterly_crossfit_tree_anchor25": (
            ("quarterly_crossfit_hist_q1", "quarterly_crossfit_extra_q1",
             "anchor_multiscale_locked"), (.375, .375, .25)),
    }
    # q2 has no score in 2023Q1 because too little post-reset fit history is
    # available; keep it as a standalone model rather than silently combining
    # non-identical calibration rows with the anchor.
    for base in ("quarterly_crossfit_hist_q1", "quarterly_crossfit_extra_q1"):
        for weight in (.25, .50):
            ensembles[f"{base}_anchor{int(weight * 100)}"] = (
                (base, "anchor_multiscale_locked"), (1.0 - weight, weight),
            )
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
        policy = (float(row.rate), int(row.rolling) or None, int(row.cooldown))
        item = _evaluate(outputs[row.candidate], (2025,), *policy,
                         y, benefit, dates, currencies)
        item.update({
            "candidate": row.candidate, "stage1_rate": policy[0],
            "stage1_rolling": policy[1] or 0, "stage1_cooldown": policy[2],
        })
        item["robustness"] = min(item["lift"], item["corridor_lift_min"])
        item["clears_1p30_gate"] = bool(
            item["lift"] >= 1.30 and .90 <= item["frequency"] <= 2.10
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
        item.update({"candidate": row.candidate,
                     "passed_2025": bool(row.clears_1p30_gate)})
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
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.heldout_start)
    ))
    if not chronology_ok:
        raise AssertionError("cross-fit chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "next_rate_feature": False, "quarterly_refit": True,
        "heldout_calibration_quarters": [1, 2],
        "score_calibration_rows_excluded_from_fit": True,
        "screen": 2024, "confirmation": 2025, "audit": 2026,
        "quarter_frequency_gate": [.30, 2.50],
        "winner": winner, "chronology_ok": chronology_ok,
        "pristine_holdout_available": False,
        "n_candidate_architectures": len(outputs),
        "n_policies_per_architecture": len(RATES) * len(POLICIES),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = ["candidate", "frequency", "lift", "forward_benefit_bps",
               "corridor_lift_min", "quarter_frequency_min",
               "quarter_frequency_max", "clears_1p30_gate"]
    print("\n2025 CONFIRMATION\n" + gate[columns].to_string(index=False))
    print("\n2026 AUDIT\n" + audit[[
        "candidate", "frequency", "lift", "forward_benefit_bps",
        "corridor_lift_min", "quarter_frequency_min", "quarter_frequency_max",
        "passed_2025",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
