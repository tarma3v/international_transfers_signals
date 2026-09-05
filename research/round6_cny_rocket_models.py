"""Packet-BG fixed random-convolution classifiers over causal CNY paths."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_rocket_features import (
    KERNELS_PER_LENGTH,
    LENGTHS,
    SEED,
    build_rocket_features,
    causality_check,
)
from research.round6_cny_waveform_features import build_waveform_features
from research.round6_moex_features import load_moex_history
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import (
    Spec,
    _bootstrap,
    _breakdown,
    _evaluate,
    prequential_scores,
)


OUT = Path("results/research/round6/cny_rocket")
PRIMARY = Path("results/research/round6/cny_consensus/outputs.pkl")
WAVE = Path("results/research/round6/cny_waveform/outputs.pkl")
ORDER = (
    "rocket_logit",
    "rocket_logit_stale20",
    "wave_logit",
    "rocket50_wave_extra50",
    "primary75_rocket25",
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    history, digest = load_moex_history()
    rocket, rocket_names = build_rocket_features(index, history)
    wave, wave_names = build_waveform_features(index, history)
    if not causality_check(index, history):
        raise AssertionError("random-convolution causality check failed")
    currency_columns = np.asarray([
        i for i, name in enumerate(names) if name.startswith("currency_")
    ], dtype=int)
    transparent = (
        "pct_range_30", "pct_range_90", "pct_range_180",
        "ret_1", "ret_5", "ret_20",
    )
    transparent_columns = np.asarray([names.index(name) for name in transparent])
    path = np.column_stack([rocket, wave])
    static = np.column_stack([X[:, currency_columns], X[:, transparent_columns]])
    matrices = {
        "rocket_logit": np.column_stack([path, static]),
        "rocket_logit_stale20": np.column_stack([
            delayed_by_currency(path, index, rows=20), static,
        ]),
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

    outputs, logs = {}, []
    for name in ("rocket_logit", "rocket_logit_stale20"):
        score, part = prequential_scores(
            Spec(name, "logit", name), matrices[name], y, dates, reach,
        )
        outputs[name] = _outputs(score, y, dates)
        logs.extend(part)
    with WAVE.open("rb") as handle:
        wave_outputs = pickle.load(handle)
    outputs["wave_logit"] = wave_outputs["wave_logit"]
    outputs["rocket50_wave_extra50"] = combine_causal(
        [outputs["rocket_logit"], wave_outputs["wave_extra"]],
        (.5, .5), dates, currencies,
    )
    with PRIMARY.open("rb") as handle:
        primary = pickle.load(handle)["logit50_extra50"]
    outputs["primary75_rocket25"] = combine_causal(
        [primary, outputs["rocket_logit"]], (.75, .25), dates, currencies,
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
        y, dates, currencies, valid, masks, "cny_rocket_2025_2026",
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
    if not chronology_ok:
        raise AssertionError("random-convolution model used unresolved labels")
    aligned = results[
        (results.candidate == "rocket_logit")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    stale = results[
        (results.candidate == "rocket_logit_stale20")
        & (results.period == "combined_2025_2026")
    ].iloc[0]
    fresh_acceptance = bool(
        aligned.lift > stale.lift
        and aligned.corridor_lift_min > stale.corridor_lift_min
    )
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "BG",
        "variants": ORDER,
        "fixed_policy": POLICY,
        "seed": SEED,
        "kernel_lengths": LENGTHS,
        "kernels_per_length": KERNELS_PER_LENGTH,
        "kernel_features": ("maximum response", "proportion positive"),
        "n_rocket_features": len(rocket_names),
        "n_waveform_features": len(wave_names),
        "stale_control_rows_per_currency": 20,
        "payload_sha256": digest,
        "asof_rule": "TRADEDATE < signal_date",
        "future_market_corruption_check": True,
        "all_training_labels_resolved": chronology_ok,
        "aligned_beats_stale_lift_and_min_currency": fresh_acceptance,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "predeclared_order", "candidate", "period", "frequency", "lift",
        "forward_benefit_bps", "corridor_lift_min", "quarter_frequency_min",
        "robustness",
    ]].sort_values(["period", "predeclared_order"]).to_string(index=False))
    print(f"\nAligned random convolution accepted as fresh: {fresh_acceptance}")


if __name__ == "__main__":
    main()
