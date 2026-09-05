"""Packet-BM low-dose global residual boosting over the frozen CNY primary."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import regime_matrix, row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate


OUT = Path("results/research/round6/cny_residual_boost")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
WAVE = Path("results/research/round6/cny_waveform/outputs.pkl")
ROCKET = Path("results/research/round6/cny_rocket/outputs.pkl")
SEED = 20260905
RANK_WINDOW = 250
RANK_MIN = 20
RESIDUAL_WEIGHT = .25
ORDER = (
    "logit50_extra50",
    "calibrated_primary",
    "residual_hist25",
    "residual_hist_stale20_25",
    "residual_extra25",
    "residual_hist50_extra50",
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _base_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.05, max_iter=3000, random_state=SEED),
    )


def _residual_model(kind):
    if kind == "hist":
        return HistGradientBoostingRegressor(
            max_iter=150, learning_rate=.03, max_leaf_nodes=5,
            min_samples_leaf=100, l2_regularization=30.0,
            random_state=SEED,
        )
    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=40,
            max_features=.65, n_jobs=1, random_state=SEED,
        )
    raise KeyError(kind)


def prequential_scores(base_matrix, matrices, y, dates, reach, verbose=True):
    scores = {
        "calibrated_primary": np.full(len(y), np.nan, dtype=float),
        "residual_hist25": np.full(len(y), np.nan, dtype=float),
        "residual_hist_stale20_25": np.full(len(y), np.nan, dtype=float),
        "residual_extra25": np.full(len(y), np.nan, dtype=float),
    }
    logs = []
    finite = np.all(np.isfinite(base_matrix), axis=1)
    for matrix in matrices.values():
        finite &= np.all(np.isfinite(matrix), axis=1)
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
            raise AssertionError("unresolved label admitted to residual booster")
        base = _base_model()
        base.fit(base_matrix[rows], y[rows])
        train_base = base.predict_proba(base_matrix[rows])[:, 1]
        test_base = base.predict_proba(base_matrix[target])[:, 1]
        scores["calibrated_primary"][target] = test_base
        residual = y[rows] - train_base
        for candidate, kind, matrix_name in (
            ("residual_hist25", "hist", "aligned"),
            ("residual_hist_stale20_25", "hist", "stale"),
            ("residual_extra25", "extra", "aligned"),
        ):
            model = _residual_model(kind)
            model.fit(matrices[matrix_name][rows], residual)
            correction = model.predict(matrices[matrix_name][target])
            scores[candidate][target] = test_base + RESIDUAL_WEIGHT * correction
            logs.append({
                "candidate": candidate,
                "quarter": str(start),
                "n_train": len(rows),
                "last_resolved": str(max(reach[rows])),
                "n_features": matrices[matrix_name].shape[1],
                "residual_weight": RESIDUAL_WEIGHT,
            })
        if verbose:
            print(
                f"  residual models              quarter={start} "
                f"train={len(rows):5d} features={matrices['aligned'].shape[1]:3d}",
                flush=True,
            )
    return scores, logs


def outcome_causality_check(base_matrix, matrices, y, dates, reach):
    cutoff = dt.date(2025, 6, 30)
    original, _ = prequential_scores(
        base_matrix, matrices, y, dates, reach, verbose=False,
    )
    changed_y = y.copy()
    unresolved = np.asarray([
        np.isfinite(value) and reach[row] > cutoff
        for row, value in enumerate(y)
    ])
    changed_y[unresolved] = 1.0 - changed_y[unresolved]
    changed, _ = prequential_scores(
        base_matrix, matrices, changed_y, dates, reach, verbose=False,
    )
    past = dates <= cutoff
    for candidate in original:
        available = past & np.isfinite(original[candidate])
        if not np.array_equal(
            original[candidate][available], changed[candidate][available],
        ):
            raise AssertionError(
                f"unresolved future outcome changed past {candidate} score"
            )
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
    currency = X[:, currency_columns]
    static = np.column_stack([currency, X[:, transparent_columns]])
    aligned = regime_matrix(ranked[:, 0], ranked[:, 1:], wave, static)
    stale = regime_matrix(
        ranked[:, 0],
        delayed_by_currency(ranked[:, 1:], index, rows=20),
        delayed_by_currency(wave, index, rows=20),
        static,
    )
    base_matrix = np.column_stack([ranked[:, 0], currency])
    matrices = {"aligned": aligned, "stale": stale}
    if not outcome_causality_check(base_matrix, matrices, y, dates, reach):
        raise AssertionError("residual outcome causality check failed")
    raw_outputs, logs = prequential_scores(
        base_matrix, matrices, y, dates, reach,
    )
    outputs = {
        "logit50_extra50": primary_output,
        **{name: _outputs(score, y, dates) for name, score in raw_outputs.items()},
    }
    outputs["residual_hist50_extra50"] = combine_causal(
        [outputs["residual_hist25"], outputs["residual_extra25"]],
        (.5, .5), dates, currencies,
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
        y, dates, currencies, valid, masks, "cny_residual_boost_2025_2026",
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
        (results.candidate == "residual_hist25")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    stale_row = results[
        (results.candidate == "residual_hist_stale20_25")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    fresh = bool(
        aligned_row.lift > stale_row.lift
        and aligned_row.corridor_lift_min > stale_row.corridor_lift_min
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BM",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "training_start": "2023-01-01",
        "rank_window": RANK_WINDOW,
        "rank_minimum": RANK_MIN,
        "base_logit_c": .05,
        "residual_weight": RESIDUAL_WEIGHT,
        "hist_residual": {
            "max_iter": 150, "learning_rate": .03, "max_leaf_nodes": 5,
            "min_samples_leaf": 100, "l2_regularization": 30.0,
        },
        "extra_residual": {
            "n_estimators": 400, "max_depth": 6,
            "min_samples_leaf": 40, "max_features": .65,
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
        raise AssertionError("residual training chronology failed")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print(f"\nAligned residual HistGB accepted as fresh: {fresh}")


if __name__ == "__main__":
    main()
