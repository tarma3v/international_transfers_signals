"""Packet-BI causal regime stack over resolved errors of fixed CNY experts."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_error_regime")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
WAVE = Path("results/research/round6/cny_waveform/outputs.pkl")
ROCKET = Path("results/research/round6/cny_rocket/outputs.pkl")
SEED = 20260905
RANK_WINDOW = 250
RANK_MIN = 20
ORDER = (
    "regime_logit",
    "regime_hist",
    "regime_hist_stale20",
    "primary75_regime_logit25",
    "primary75_regime_hist25",
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def row_scores(output, n_rows):
    scores = np.full(n_rows, np.nan, dtype=float)
    # Calibration scores can be re-normalized when they are carried into the
    # following year's output.  Use them only to seed 2023, then let every row's
    # own-year test score be authoritative.
    for year in sorted(output):
        rows = np.asarray(output[year]["calib_idx"], dtype=int)
        values = np.asarray(output[year]["calib_score"], dtype=float)
        missing = ~np.isfinite(scores[rows])
        scores[rows[missing]] = values[missing]
    for year in sorted(output):
        rows = np.asarray(output[year]["test_idx"], dtype=int)
        scores[rows] = np.asarray(output[year]["test_score"], dtype=float)
    return scores


def regime_matrix(primary_rank, auxiliary_ranks, wave, static):
    ranks = np.column_stack([primary_rank, auxiliary_ranks])
    disagreements = np.column_stack([
        np.abs(ranks[:, 0] - ranks[:, 1]),
        np.abs(ranks[:, 0] - ranks[:, 2]),
        np.abs(ranks[:, 1] - ranks[:, 2]),
    ])
    aggregates = np.column_stack([
        np.mean(ranks, axis=1),
        np.min(ranks, axis=1),
        np.max(ranks, axis=1),
        np.std(ranks, axis=1),
    ])
    return np.column_stack([ranks, disagreements, aggregates, static, wave])


def _model(kind):
    if kind == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.02, max_iter=3000, random_state=SEED),
        )
    if kind == "hist":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.03, max_leaf_nodes=5,
            min_samples_leaf=80, l2_regularization=25.0,
            random_state=SEED,
        )
    raise KeyError(kind)


def prequential_scores(name, kind, matrix, y, dates, reach):
    scores = np.full(len(y), np.nan, dtype=float)
    logs = []
    finite = np.all(np.isfinite(matrix), axis=1)
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        train = (
            (dates >= dt.date(2023, 1, 1))
            & np.asarray([value < start for value in reach])
            & np.isfinite(y) & finite
        )
        test = (
            (dates >= start) & (dates < end)
            & np.isfinite(y) & finite
        )
        rows = np.flatnonzero(train)
        target = np.flatnonzero(test)
        if len(rows) < 700 or not len(target):
            continue
        if not all(reach[row] < start for row in rows):
            raise AssertionError("unresolved label admitted to regime stack")
        model = _model(kind)
        model.fit(matrix[rows], y[rows])
        scores[target] = model.predict_proba(matrix[target])[:, 1]
        logs.append({
            "candidate": name,
            "quarter": str(start),
            "n_train": len(rows),
            "last_resolved": str(max(reach[rows])),
            "n_features": matrix.shape[1],
        })
        print(
            f"  {name:<28} quarter={start} train={len(rows):5d} "
            f"features={matrix.shape[1]:3d}",
            flush=True,
        )
    return scores, logs


def outcome_causality_check(matrix, y, dates, reach):
    cutoff = dt.date(2025, 6, 30)
    original, _ = prequential_scores(
        "causality_original", "logit", matrix, y, dates, reach,
    )
    changed_y = y.copy()
    unresolved = np.asarray([
        np.isfinite(value) and reach[row] > cutoff
        for row, value in enumerate(y)
    ])
    changed_y[unresolved] = 1.0 - changed_y[unresolved]
    changed, _ = prequential_scores(
        "causality_corrupted", "logit", matrix, changed_y, dates, reach,
    )
    past = (dates <= cutoff) & np.isfinite(original)
    if not np.array_equal(original[past], changed[past]):
        raise AssertionError("unresolved future outcome changed a past regime score")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    primary_output = _load(PRIMARY)["logit50_extra50"]
    wave_output = _load(WAVE)["wave_extra"]
    rocket_output = _load(ROCKET)["rocket_logit"]
    raw_scores = np.column_stack([
        row_scores(primary_output, len(index)),
        row_scores(wave_output, len(index)),
        row_scores(rocket_output, len(index)),
    ])
    ranked = np.column_stack([
        causal_percentiles(
            raw_scores[:, column], dates, currencies,
            window=RANK_WINDOW, minimum=RANK_MIN,
        )
        for column in range(raw_scores.shape[1])
    ])
    history, digest = load_moex_history()
    wave, wave_names = build_waveform_features(index, history)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    transparent = (
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_5", "ret_20",
    )
    transparent_columns = np.asarray([names.index(name) for name in transparent])
    static = np.column_stack([X[:, currency_columns], X[:, transparent_columns]])
    aligned = regime_matrix(ranked[:, 0], ranked[:, 1:], wave, static)
    delayed_auxiliary = delayed_by_currency(ranked[:, 1:], index, rows=20)
    delayed_wave = delayed_by_currency(wave, index, rows=20)
    stale = regime_matrix(ranked[:, 0], delayed_auxiliary, delayed_wave, static)
    if not outcome_causality_check(aligned, y, dates, reach):
        raise AssertionError("regime outcome causality check failed")

    outputs, logs = {}, []
    for name, kind, matrix in (
        ("regime_logit", "logit", aligned),
        ("regime_hist", "hist", aligned),
        ("regime_hist_stale20", "hist", stale),
    ):
        score, part = prequential_scores(name, kind, matrix, y, dates, reach)
        outputs[name] = _outputs(score, y, dates)
        logs.extend(part)
    outputs["primary75_regime_logit25"] = combine_causal(
        [primary_output, outputs["regime_logit"]], (.75, .25), dates, currencies,
    )
    outputs["primary75_regime_hist25"] = combine_causal(
        [primary_output, outputs["regime_hist"]], (.75, .25), dates, currencies,
    )
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    training = pd.DataFrame(logs)
    training.to_csv(OUT / "training_log.csv", index=False)

    rows = []
    for candidate in ORDER:
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                outputs[candidate], years, POLICY, y, benefit, dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            item["robustness"] = min(item["lift"], item["corridor_lift_min"])
            rows.append(item)
    results = pd.DataFrame(rows)
    results["predeclared_order"] = results.candidate.map(
        {name: i + 1 for i, name in enumerate(ORDER)}
    )
    results.to_csv(OUT / "matched_results.csv", index=False)
    screen = results[results.period == "screen_2024"].copy()
    bootstrap, masks, valid = _bootstrap(
        screen, outputs, (2025, 2026), y, benefit, dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_2025_2026.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "cny_error_regime_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    breakdown = []
    for candidate in ORDER:
        breakdown.extend(_breakdown(
            candidate, outputs[candidate], (2025, 2026), POLICY,
            y, benefit, dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_2025_2026.csv", index=False)
    chronology_ok = bool(np.all(
        pd.to_datetime(training.last_resolved) < pd.to_datetime(training.quarter)
    ))
    aligned_row = results[
        (results.candidate == "regime_hist")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    stale_row = results[
        (results.candidate == "regime_hist_stale20")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    fresh = bool(
        aligned_row.lift > stale_row.lift
        and aligned_row.corridor_lift_min > stale_row.corridor_lift_min
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BI",
        "experts": ("logit50_extra50", "wave_extra", "rocket_logit"),
        "variants": ORDER,
        "fixed_policy": POLICY,
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MIN,
        "training_start": "2023-01-01",
        "logit_c": .02,
        "histogram_booster": {
            "max_iter": 200, "learning_rate": .03, "max_leaf_nodes": 5,
            "min_samples_leaf": 80, "l2_regularization": 25.0,
        },
        "stale_control_rows_per_currency": 20,
        "waveform_features": wave_names,
        "payload_sha256": digest,
        "future_outcome_corruption_check": True,
        "all_training_labels_resolved": chronology_ok,
        "aligned_hist_beats_stale_lift_and_min_currency": fresh,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not chronology_ok:
        raise AssertionError("regime training chronology failed")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print(f"\nAligned histogram regime accepted as fresh: {fresh}")


if __name__ == "__main__":
    main()
