"""Packet-CV: frozen pre-2024 calendar priors as a low-dose expert."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.data import CORRIDORS
from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_multiobjective_blend import combine_causal
from research.round6_uzbek_central_bank_models import (
    _choose, _forward, horizon_rows, summarize,
)


OUT = Path("results/research/round6/calendar_prior")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
TRAIN_START = dt.date(2016, 1, 1)
FREEZE = dt.date(2024, 1, 1)


def calendar_matrix(index):
    rows = []
    names = None
    for currency, _position, day in index:
        dow = day.weekday()
        month = day.month - 1
        dom_bin = min((day.day - 1) // 7, 3)
        days_in_month = (dt.date(day.year + (day.month == 12), day.month % 12 + 1, 1) - dt.timedelta(days=1)).day
        end_distance = days_in_month - day.day
        currency_id = CORRIDORS.index(currency)
        values = []
        row_names = []
        for label, value, size in (
            ("currency", currency_id, len(CORRIDORS)),
            ("dow", dow, 7), ("month", month, 12), ("dom_bin", dom_bin, 4),
        ):
            values.extend(float(value == i) for i in range(size))
            row_names.extend(f"{label}_{i}" for i in range(size))
        angle_dow = 2 * np.pi * dow / 7.0
        angle_dom = 2 * np.pi * (day.day - 1) / days_in_month
        angle_month = 2 * np.pi * month / 12.0
        values.extend((
            np.sin(angle_dow), np.cos(angle_dow),
            np.sin(angle_dom), np.cos(angle_dom),
            np.sin(angle_month), np.cos(angle_month),
            float(end_distance), float(end_distance <= 2),
            float(day.day <= 3), float(day.day in (10, 15, 20, 25)),
            float(month in (11, 0)), float(month in (2, 5, 8, 11) and end_distance <= 5),
        ))
        row_names.extend((
            "dow_sin", "dow_cos", "dom_sin", "dom_cos",
            "month_sin", "month_cos", "days_to_month_end",
            "month_end_3d", "month_start_3d", "payday_exact",
            "new_year_season", "quarter_end_5d",
        ))
        for weekday in range(7):
            values.extend(float(currency_id == c and dow == weekday) for c in range(len(CORRIDORS)))
            row_names.extend(f"currency_{c}_dow_{weekday}" for c in range(len(CORRIDORS)))
        for bin_id in range(4):
            values.extend(float(currency_id == c and dom_bin == bin_id) for c in range(len(CORRIDORS)))
            row_names.extend(f"currency_{c}_dom_bin_{bin_id}" for c in range(len(CORRIDORS)))
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("calendar schema changed")
    return np.asarray(rows, dtype=np.float32), names or []


def aggregate(probabilities, prefix):
    clipped = np.clip(probabilities, 1e-6, 1.0)
    weights = np.asarray((.10, .15, .25, .25, .25))
    return {
        f"{prefix}_h1": probabilities[:, 0],
        f"{prefix}_h5": probabilities[:, HORIZONS.index(5)],
        f"{prefix}_minimum": probabilities.min(axis=1),
        f"{prefix}_geometric": np.exp(np.mean(np.log(clipped), axis=1)),
        f"{prefix}_weighted": np.exp(np.sum(np.log(clipped) * weights, axis=1)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    labels = np.column_stack([targets[f"fav_h{h}"] for h in HORIZONS])
    reach = target_reach_dates(index, series, 20)
    matrix, names = calendar_matrix(index)
    train = np.flatnonzero(
        (dates >= TRAIN_START) & (dates < FREEZE)
        & np.asarray([value < FREEZE for value in reach])
        & np.all(np.isfinite(labels), axis=1)
    )
    scores = {}
    coefficients = []
    for c_value in (.005, .02, .10):
        probabilities = np.full_like(labels, np.nan, dtype=float)
        for column, h in enumerate(HORIZONS):
            model = LogisticRegression(
                C=c_value, max_iter=4000, random_state=20260905,
            ).fit(matrix[train], labels[train, column])
            probabilities[:, column] = model.predict_proba(matrix)[:, 1]
            coefficients.extend({
                "c": c_value, "horizon": h, "feature": name,
                "coefficient": float(value),
            } for name, value in zip(names, model.coef_[0]))
        scores.update(aggregate(probabilities, f"calendar_c{str(c_value).replace('.', 'p')}"))
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    raw_outputs = {
        name: _outputs(score, targets["fav_h5"], dates)
        for name, score in scores.items()
    }
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    raw_screen = horizon_rows(
        raw_outputs, (2024,), targets, forwards, dates, currencies,
    )
    raw_summary = summarize(raw_screen)
    selected_raw = _choose(raw_summary)
    with INCUMBENT_PATH.open("rb") as handle:
        incumbent = pickle.load(handle)[INCUMBENT]
    finalists = {"incumbent": incumbent, "calendar_selected": raw_outputs[selected_raw]}
    for weight in (.025, .05, .10, .15, .20, .30):
        finalists[f"incumbent{int(round((1-weight)*1000)):03d}_calendar{int(round(weight*1000)):03d}"] = combine_causal(
            [incumbent, raw_outputs[selected_raw]], (1.0 - weight, weight),
            dates, currencies,
        )
    finalist_screen = horizon_rows(
        finalists, (2024,), targets, forwards, dates, currencies,
    )
    finalist_summary = summarize(finalist_screen)
    selected_finalist = _choose(finalist_summary)
    raw_screen.to_csv(OUT / "raw_screen_2024_by_horizon.csv", index=False)
    raw_summary.to_csv(OUT / "raw_screen_2024_summary.csv", index=False)
    finalist_screen.to_csv(OUT / "finalist_screen_2024_by_horizon.csv", index=False)
    finalist_summary.to_csv(OUT / "finalist_screen_2024_summary.csv", index=False)

    comparison = {"incumbent": incumbent, "selected": finalists[selected_finalist]}
    later_parts = []
    for period, years in (
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        part = horizon_rows(comparison, years, targets, forwards, dates, currencies)
        part["period"] = period
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    later.to_csv(OUT / "later_by_horizon.csv", index=False)
    later_summary = summarize(later[later.period == "combined_2025_2026"])
    later_summary.to_csv(OUT / "later_summary.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CV", "train_start": str(TRAIN_START), "freeze": str(FREEZE),
        "last_training_target_reach_before_freeze": True,
        "feature_count": matrix.shape[1], "training_rows": len(train),
        "regularization_c": [.005, .02, .10],
        "target_horizons": HORIZONS, "raw_selected": selected_raw,
        "finalist_selected": selected_finalist,
        "selection_period": 2024,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected raw:", selected_raw)
    print("Selected finalist:", selected_finalist)
    print("\nRAW\n" + raw_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nSCREEN\n" + finalist_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))


if __name__ == "__main__":
    main()
