"""Packet-DE: causal local-linear state-space scores for target currencies."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import Series
from ml.targets import HORIZONS, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_crossbank_consensus import INCUMBENT, INCUMBENT_PATH
from research.round6_cny_decomposition import POLICY
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/target_state_space")
GAINS = ((.10, .01), (.20, .03), (.40, .08))
RESET = dt.date(2022, 2, 24)
SCALE_WINDOW = 120
MIN_SCALE_HISTORY = 20
BLEND_WEIGHTS = (.05, .10, .20, .30)
SCORE_KINDS = ("next_gap", "negative_innovation", "positive_slope", "rebound")


def _filter(values, dates, alpha, beta, reset):
    scores = {name: np.zeros(len(values), dtype=float) for name in SCORE_KINDS}
    level = slope = None
    innovations = []
    reset_done = not reset
    for i, (raw, day) in enumerate(zip(values, dates)):
        observed = float(np.log(raw))
        if reset and not reset_done and day >= RESET:
            level = observed
            slope = 0.0
            innovations = []
            reset_done = True
            continue
        if level is None:
            level = observed
            slope = 0.0
            continue
        predicted = level + slope
        innovation = observed - predicted
        if len(innovations) >= MIN_SCALE_HISTORY:
            reference = np.asarray(innovations[-SCALE_WINDOW:], dtype=float)
            center = float(np.median(reference))
            mad = float(np.median(np.abs(reference - center)))
            scale = max(1.4826 * mad, 1e-5)
        else:
            scale = max(float(np.std(innovations)), 1e-5) if innovations else 1e-5
        level = predicted + alpha * innovation
        slope = slope + beta * innovation
        negative_innovation = -innovation / scale
        positive_slope = slope / scale
        next_gap = (level + slope - observed) / scale
        scores["next_gap"][i] = next_gap
        scores["negative_innovation"][i] = negative_innovation
        scores["positive_slope"][i] = positive_slope
        scores["rebound"][i] = .5 * (negative_innovation + positive_slope)
        innovations.append(innovation)
    return scores


def build_state_features(series, index):
    by_currency = {}
    names = []
    for alpha, beta in GAINS:
        tag = f"a{int(alpha*100):02d}_b{int(beta*100):02d}"
        for reset in (False, True):
            regime = "reset2022" if reset else "full"
            for kind in SCORE_KINDS:
                names.append(f"{regime}_{tag}_{kind}")
    for currency, item in series.items():
        columns = {}
        for alpha, beta in GAINS:
            tag = f"a{int(alpha*100):02d}_b{int(beta*100):02d}"
            for reset in (False, True):
                regime = "reset2022" if reset else "full"
                filtered = _filter(item.values, item.dates, alpha, beta, reset)
                for kind, values in filtered.items():
                    columns[f"{regime}_{tag}_{kind}"] = values
        by_currency[currency] = columns
    matrix = np.asarray([
        [by_currency[currency][name][position] for name in names]
        for currency, position, _day in index
    ], dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite target state-space feature")
    return matrix, names


def causality_check(series, index, cutoff=dt.date(2025, 6, 30)):
    full, names = build_state_features(series, index)
    changed = {}
    for currency, item in series.items():
        values = item.values.copy()
        future = item.dates > cutoff
        values[future] *= np.linspace(2.0, 20.0, int(future.sum()))
        changed[currency] = Series(currency, item.dates.copy(), values)
    altered, altered_names = build_state_features(changed, index)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future target changed a past state-space feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future corruption did not affect future state scores")
    return True


def _load_incumbent():
    with INCUMBENT_PATH.open("rb") as handle:
        return pickle.load(handle)[INCUMBENT]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    matrix, names = build_state_features(series, index)
    causality_check(series, index)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]
    raw_outputs = {
        name: _outputs(matrix[:, i].astype(float), y5, dates)
        for i, name in enumerate(names)
    }
    raw_screen = horizon_rows(raw_outputs, (2024,), targets, forwards, dates, currencies)
    raw_summary = summarize(raw_screen)
    selected_raw = _choose(raw_summary)

    incumbent = _load_incumbent()
    finalists = {"incumbent": incumbent, "state_selected": raw_outputs[selected_raw]}
    for weight in BLEND_WEIGHTS:
        name = f"incumbent{int((1-weight)*100)}_state{int(weight*100)}"
        finalists[name] = combine_causal(
            [incumbent, raw_outputs[selected_raw]], (1.0 - weight, weight),
            dates, currencies,
        )
    finalist_screen = horizon_rows(finalists, (2024,), targets, forwards, dates, currencies)
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

    h5_rows = []
    for candidate, output in finalists.items():
        for period, years in (
            ("screen_2024", (2024,)),
            ("retrospective_2025", (2025,)),
            ("retrospective_2026", (2026,)),
            ("combined_2025_2026", (2025, 2026)),
        ):
            item = _evaluate(output, years, POLICY, y5, forwards[5], dates, currencies)
            item.update({"candidate": candidate, "period": period, **POLICY})
            h5_rows.append(item)
    h5 = pd.DataFrame(h5_rows)
    h5.to_csv(OUT / "standard_h5_results.csv", index=False)
    with (OUT / "outputs.pkl").open("wb") as handle:
        pickle.dump(finalists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], finalists, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "target_state_space_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in finalists.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DE", "fixed_policy": POLICY,
        "gains": GAINS, "score_kinds": SCORE_KINDS,
        "reset_date": RESET, "scale_window": SCALE_WINDOW,
        "minimum_scale_history": MIN_SCALE_HISTORY,
        "raw_candidates": names, "blend_weights": BLEND_WEIGHTS,
        "selection_period": 2024, "raw_selected": selected_raw,
        "finalist_selected": selected_finalist,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "feature_information": "current and earlier target publications only",
        "physical_future_corruption_check": True,
        "next_cbr_rate_used": False,
        "later_period_status": "protocol-controlled retrospective opened after 2024 selection",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected raw on 2024:", selected_raw)
    print("Selected finalist on 2024:", selected_finalist)
    print("\nRAW SCREEN\n" + raw_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nFINALIST SCREEN\n" + finalist_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
