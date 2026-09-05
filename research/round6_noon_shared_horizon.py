"""Packet-DR: shared official-horizon learner on the noon MOEX state."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _next_quarter, _outputs, _quarter_starts
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_moex_perpetual_hourly_features import (
    build_hourly_features,
    load_hourly_history,
)
from research.round6_moex_perpetual_models import _model
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/noon_shared_horizon")
NOON_CONSENSUS_PATH = Path(
    "results/research/round6/three_view_futures_consensus/outputs.pkl"
)
TRAIN_START = dt.date(2022, 5, 1)
KINDS = ("hist", "extra")
AGGREGATES = ("minimum", "geometric", "arithmetic", "conservative")
BLEND_WEIGHTS = (.25, .50)
STALE_ROWS = 20
RANK_WINDOW = 250
RANK_MINIMUM = 20
TARGET_FEATURES = (
    "pct_range_30", "pct_range_90", "pct_range_180",
    "ret_1", "ret_5", "ret_20",
)


def horizon_features(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Replicate prediction rows in row-major official-horizon order."""
    count = len(HORIZONS)
    repeated = np.repeat(matrix[rows], count, axis=0)
    one_hot = np.tile(np.eye(count, dtype=np.float32), (len(rows), 1))
    log_h = np.tile(
        np.log(np.asarray(HORIZONS, dtype=float)) / np.log(max(HORIZONS)),
        len(rows),
    )[:, None]
    return np.column_stack((repeated, one_hot, log_h))


def shared_scores(kind, matrix, labels, dates, reaches, verbose=True):
    scores = np.full((len(dates), len(HORIZONS)), np.nan, dtype=float)
    finite = np.all(np.isfinite(matrix), axis=1)
    logs = []
    eye = np.eye(len(HORIZONS), dtype=np.float32)
    log_h = np.log(np.asarray(HORIZONS, dtype=float)) / np.log(max(HORIZONS))
    for start in _quarter_starts():
        if start.year < 2024:
            continue
        end = _next_quarter(start)
        target = np.flatnonzero(
            (dates >= start) & (dates < end) & finite
        )
        if not len(target):
            continue
        train_x, train_y, admitted_reaches = [], [], []
        counts = {}
        for column, horizon in enumerate(HORIZONS):
            reach = reaches[horizon]
            train = (
                (dates >= TRAIN_START) & finite & np.isfinite(labels[:, column])
                & np.asarray([value < start for value in reach])
            )
            rows = np.flatnonzero(train)
            counts[str(horizon)] = int(len(rows))
            block = np.column_stack((
                matrix[rows],
                np.tile(eye[column], (len(rows), 1)),
                np.full((len(rows), 1), log_h[column]),
            ))
            train_x.append(block)
            train_y.append(labels[rows, column])
            admitted_reaches.extend(reach[rows])
        x_fit = np.vstack(train_x)
        y_fit = np.concatenate(train_y)
        if len(x_fit) < 5000 or len(np.unique(y_fit)) < 2:
            continue
        if not all(value < start for value in admitted_reaches):
            raise AssertionError("unresolved official-horizon label admitted")
        learner = _model(kind)
        learner.fit(x_fit, y_fit)
        probability = learner.predict_proba(
            horizon_features(matrix, target)
        )[:, 1]
        scores[target] = probability.reshape(len(target), len(HORIZONS))
        logs.append({
            "kind": kind, "quarter": str(start),
            "n_train_replicas": int(len(x_fit)), "n_test_rows": int(len(target)),
            "last_resolved": str(max(admitted_reaches)),
            "n_features": int(x_fit.shape[1]),
            "n_train_by_horizon": json.dumps(counts, sort_keys=True),
        })
        if verbose:
            print(
                f"  noon shared {kind:<5} quarter={start} "
                f"train={len(x_fit):6d} test={len(target):4d}",
                flush=True,
            )
    return scores, logs


def aggregate_scores(scores, dates, currencies):
    ranks = np.column_stack([
        causal_percentiles(
            scores[:, column], dates, currencies, RANK_WINDOW, RANK_MINIMUM,
        )
        for column in range(scores.shape[1])
    ])
    clipped = np.clip(ranks, 1e-9, 1.0)
    return {
        "minimum": np.min(ranks, axis=1),
        "geometric": np.exp(np.mean(np.log(clipped), axis=1)),
        "arithmetic": np.mean(ranks, axis=1),
        "conservative": np.mean(ranks, axis=1) - .5 * np.std(ranks, axis=1),
    }


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_hourly_history()
    hourly, hourly_names = build_hourly_features(index, history, references)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    target_columns = np.asarray([names.index(name) for name in TARGET_FEATURES])
    static = np.column_stack((X[:, currency_columns], X[:, target_columns]))
    aligned = np.column_stack((static, hourly))
    stale = np.column_stack((
        static, delayed_by_currency(hourly, index, rows=STALE_ROWS),
    ))

    targets = build_targets(series, index)
    labels = np.column_stack([targets[f"fav_h{h}"] for h in HORIZONS])
    reaches = {h: target_reach_dates(index, series, h) for h in HORIZONS}
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]

    aligned_outputs, stale_outputs, logs = {}, {}, []
    for kind in KINDS:
        fresh_scores, fresh_log = shared_scores(
            kind, aligned, labels, dates, reaches,
        )
        stale_scores, stale_log = shared_scores(
            kind, stale, labels, dates, reaches,
        )
        fresh_aggregates = aggregate_scores(fresh_scores, dates, currencies)
        stale_aggregates = aggregate_scores(stale_scores, dates, currencies)
        for aggregation in AGGREGATES:
            name = f"shared_{kind}_{aggregation}"
            aligned_outputs[name] = _outputs(
                fresh_aggregates[aggregation], y5, dates,
            )
            stale_outputs[name] = _outputs(
                stale_aggregates[aggregation], y5, dates,
            )
        logs.extend({**row, "stale": False} for row in fresh_log)
        logs.extend({**row, "stale": True} for row in stale_log)

    incumbent = _load(INCUMBENT_PATH, INCUMBENT)
    noon_consensus = _load(NOON_CONSENSUS_PATH, "selected")
    candidates = {
        "incumbent": incumbent,
        "noon_consensus": noon_consensus,
        **aligned_outputs,
    }
    matched_stale = dict(stale_outputs)
    for name, raw in aligned_outputs.items():
        for weight in BLEND_WEIGHTS:
            candidate = f"incumbent{int((1-weight)*100)}_{name}{int(weight*100)}"
            candidates[candidate] = combine_causal(
                (incumbent, raw), (1.0 - weight, weight), dates, currencies,
            )
            matched_stale[candidate] = combine_causal(
                (incumbent, stale_outputs[name]),
                (1.0 - weight, weight), dates, currencies,
            )

    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {
        "incumbent": incumbent,
        "noon_consensus": noon_consensus,
        "selected": candidates[selected],
    }
    if selected in matched_stale:
        comparison["matched_stale20"] = matched_stale[selected]
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)

    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(
            comparison, years, targets, forwards, dates, currencies,
        )
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)

    h5_rows = []
    for candidate, output in comparison.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(
                output, years, POLICY, y5, forwards[5], dates, currencies,
            )
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], comparison, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "noon_shared_horizon_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)

    chronology_ok = bool(all(
        dt.date.fromisoformat(row["last_resolved"])
        < dt.date.fromisoformat(row["quarter"])
        for row in logs
    ))
    if not chronology_ok:
        raise AssertionError("shared noon training chronology failed")
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DR", "fixed_policy": POLICY,
        "hourly_source_payload_sha256": digest,
        "decision_time": "12:00:00 Europe/Moscow",
        "strict_hourly_asof": "candle end < signal_date 12:00:00",
        "hourly_feature_count": len(hourly_names),
        "target_features": TARGET_FEATURES,
        "horizons": HORIZONS, "models": KINDS,
        "aggregates": AGGREGATES, "blend_weights": BLEND_WEIGHTS,
        "rank_window": RANK_WINDOW, "rank_minimum": RANK_MINIMUM,
        "stale_rows": STALE_ROWS, "training_start": TRAIN_START,
        "all_horizon_labels_resolved_before_refit": chronology_ok,
        "selection_period": 2024, "selected": selected,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("\nSCREEN TOP\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).head(20).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
