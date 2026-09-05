"""Packet-BK full 2017--2026 lifecycle transport for CNY path challengers."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_cny_lifecycle import YEARS, _stitch
from research.round6_cny_pre2022 import CNY_START, next_quarter
from research.round6_cny_rocket_features import build_rocket_features
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    _bootstrap,
    _breakdown,
    _evaluate,
    _model,
)


OUT = Path("results/research/round6/cny_rocket_lifecycle")
PRIMARY = Path("results/research/round6/cny_lifecycle/outputs.pkl")
ROCKET = Path("results/research/round6/cny_rocket/outputs.pkl")
REGIME = Path("results/research/round6/cny_error_regime/outputs.pkl")
HANDOFF = dt.date(2024, 1, 1)
PRE_YEARS = tuple(range(2017, 2024))
LATER_YEARS = tuple(range(2024, 2027))
ORDER = (
    "primary_resolved2000",
    "rocket_resolved2000",
    "primary75_rocket25_lifecycle",
    "primary_then_regime2024",
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quarter_starts():
    return [
        dt.date(year, month, 1)
        for year in range(2016, 2024) for month in (1, 4, 7, 10)
    ]


def expanding_scores(matrix, y, dates, reach):
    score = np.full(len(y), np.nan, dtype=float)
    logs = []
    for start in quarter_starts():
        end = next_quarter(start)
        train = (
            (dates >= CNY_START)
            & np.asarray([value < start for value in reach])
            & np.isfinite(y)
        )
        test = (dates >= start) & (dates < end) & np.isfinite(y)
        rows = np.flatnonzero(train)
        target = np.flatnonzero(test)
        if len(rows) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError("unresolved label admitted to expanding rocket")
        model = _model("logit")
        model.fit(matrix[rows], y[rows])
        score[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": "rocket_expanding",
            "quarter": str(start),
            "n_train": len(rows),
            "first_train": str(min(dates[rows])),
            "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1],
        })
        print(
            f"  rocket_expanding            quarter={start} "
            f"train={len(rows):5d} features={matrix.shape[1]:3d}",
            flush=True,
        )
    return score, logs


def yearly_outputs(score, y, dates):
    result = {}
    for year in PRE_YEARS:
        calibration = (
            (dates >= dt.date(year - 1, 1, 1))
            & (dates < dt.date(year, 1, 1))
            & np.isfinite(score) & np.isfinite(y)
        )
        test = (
            np.asarray([day.year == year for day in dates])
            & np.isfinite(score) & np.isfinite(y)
        )
        ca, te = np.flatnonzero(calibration), np.flatnonzero(test)
        result[year] = {
            "calib_idx": ca,
            "test_idx": te,
            "calib_score": score[ca],
            "test_score": score[te],
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    rocket, rocket_names = build_rocket_features(index, history)
    wave, wave_names = build_waveform_features(index, history)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    transparent = (
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_5", "ret_20",
    )
    transparent_columns = np.asarray([names.index(name) for name in transparent])
    matrix = np.column_stack([
        rocket,
        wave,
        X[:, currency_columns],
        X[:, transparent_columns],
    ])
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    expanding_score, logs = expanding_scores(matrix, y, dates, reach)
    expanding = yearly_outputs(expanding_score, y, dates)
    reset = _load(ROCKET)["rocket_logit"]
    reset_later = {year: reset[year] for year in LATER_YEARS}
    rocket_lifecycle = _stitch(expanding, reset_later)

    primary = _load(PRIMARY)["resolved2000_handoff"]
    regime = _load(REGIME)["primary75_regime_logit25"]
    regime_later = {year: regime[year] for year in LATER_YEARS}
    primary_pre = {year: primary[year] for year in PRE_YEARS}
    outputs = {
        "primary_resolved2000": primary,
        "rocket_resolved2000": rocket_lifecycle,
        "primary75_rocket25_lifecycle": combine_causal(
            [primary, rocket_lifecycle], (.75, .25), dates, currencies,
        ),
        "primary_then_regime2024": _stitch(primary_pre, regime_later),
    }
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate in ORDER:
        for year in YEARS:
            item = _evaluate(
                outputs[candidate], (year,), POLICY,
                y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": str(year), **POLICY})
            rows.append(item)
        item = _evaluate(
            outputs[candidate], YEARS, POLICY,
            y, benefit, dates, currencies,
        )
        item.update({"candidate": candidate, "period": "2017_2026", **POLICY})
        rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    combined = results[results.period == "2017_2026"].copy()
    bootstrap, masks, valid = _bootstrap(
        combined, outputs, YEARS, y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2017_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "rocket_lifecycle_2017_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], YEARS, POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2017_2026.csv", index=False)

    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    control = results[
        (results.candidate == "primary_resolved2000")
        & (results.period == "2017_2026")
    ].iloc[0]
    feasibility = {}
    for candidate in ORDER[1:]:
        yearly = results[
            (results.candidate == candidate) & results.period.isin(map(str, YEARS))
        ]
        combined_row = results[
            (results.candidate == candidate) & (results.period == "2017_2026")
        ].iloc[0]
        gates = {
            "combined_lift_above_primary": bool(combined_row.lift > control.lift),
            "all_annual_lift_at_least_1_30": bool(yearly.lift.ge(1.30).all()),
            "all_annual_rate_between_1_and_2": bool(yearly.frequency.between(1, 2).all()),
            "minimum_currency_lift_at_least_1_30": bool(
                combined_row.corridor_lift_min >= 1.30
            ),
            "minimum_quarter_rate_at_least_0_90": bool(
                combined_row.quarter_frequency_min >= .90
            ),
        }
        gates["lifecycle_feasible"] = bool(all(gates.values()))
        feasibility[candidate] = gates
    source_hashes = {str(path): _digest(path) for path in (PRIMARY, ROCKET, REGIME)}
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BK",
        "years": YEARS,
        "variants": ORDER,
        "fixed_policy": POLICY,
        "handoff_rule": "expanding through 2023; post-2022 reset from 2024-01-01",
        "resolved2000_handoff_date": str(HANDOFF),
        "rocket_features": len(rocket_names) + len(wave_names),
        "payload_sha256": digest,
        "source_output_sha256": source_hashes,
        "all_training_labels_resolved": chronology_ok,
        "all_years_retained": True,
        "feasibility_gates": feasibility,
        "later_period_status": "retrospective composition of frozen causal scores",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("expanding rocket chronology failed")
    columns = [
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "year_lift_min", "year_frequency_min",
        "corridor_lift_min", "quarter_frequency_min",
    ]
    print(results[columns].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print("\nFEASIBILITY\n" + json.dumps(feasibility, indent=2))


if __name__ == "__main__":
    main()
