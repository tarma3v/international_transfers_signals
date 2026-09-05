"""Packet-DC: within-source revisions of seven shadow-RUB cross-rates.

The packet differs each bank against its own earlier value before aggregating,
so changing source availability cannot masquerade as a market move.  Formula
selection uses 2024 only; 2025-26 is opened once after the finalist is fixed.
"""
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
from research.round6_crossbank_consensus import (
    INCUMBENT,
    INCUMBENT_PATH,
    MAX_AGE_DAYS,
    SOURCES,
    _source_bases,
)
from research.round6_cny_decomposition import POLICY
from research.round6_broad_cbr_features import load_broad_features
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/crossbank_revision_dynamics")
LAGS = (1, 5, 20)
BLEND_WEIGHTS = (.05, .10, .20, .30)


def build_revision_features(index, references, sources):
    """Difference within source, then aggregate across sources."""
    days = sorted({row[2] for row in index})
    source_names = tuple(sources)
    usd = np.full((len(days), len(source_names)), np.nan)
    cny = np.full_like(usd, np.nan)
    availability = []
    for i, day in enumerate(days):
        present = []
        for j, name in enumerate(source_names):
            usd[i, j], cny[i, j], _age = _source_bases(
                sources[name], references, day,
            )
            if np.isfinite(usd[i, j]) and np.isfinite(cny[i, j]):
                present.append(name)
        availability.append({
            "date": day,
            "source_count": len(present),
            "source_names": ",".join(present),
        })

    consensus = .5 * (usd + cny)
    rows, names = [], []
    raw_by_name = {}
    for lag in LAGS:
        lag_usd = np.full_like(usd, np.nan)
        lag_cny = np.full_like(cny, np.nan)
        lag_consensus = np.full_like(consensus, np.nan)
        if lag < len(days):
            lag_usd[lag:] = usd[:-lag]
            lag_cny[lag:] = cny[:-lag]
            lag_consensus[lag:] = consensus[:-lag]
        delta_usd = usd - lag_usd
        delta_cny = cny - lag_cny
        delta = consensus - lag_consensus

        median_change = np.zeros(len(days))
        jointly_positive = np.zeros(len(days))
        jointly_negative = np.zeros(len(days))
        dispersion = np.zeros(len(days))
        for i in range(len(days)):
            paired = np.isfinite(delta[i]) & np.isfinite(delta_usd[i]) & np.isfinite(delta_cny[i])
            if int(paired.sum()) < 3:
                continue
            values = delta[i, paired]
            median_change[i] = float(np.median(values))
            jointly_positive[i] = float(np.mean(
                (delta_usd[i, paired] > 0.0) & (delta_cny[i, paired] > 0.0)
            ))
            jointly_negative[i] = float(np.mean(
                (delta_usd[i, paired] < 0.0) & (delta_cny[i, paired] < 0.0)
            ))
            center = median_change[i]
            dispersion[i] = float(np.median(np.abs(values - center)))

        raw_by_name[f"revision_{lag}"] = median_change
        raw_by_name[f"negative_revision_{lag}"] = -median_change
        raw_by_name[f"joint_positive_breadth_{lag}"] = jointly_positive
        raw_by_name[f"joint_negative_breadth_{lag}"] = jointly_negative
        if lag == 5:
            raw_by_name["negative_revision_dispersion_5"] = -dispersion

    short = raw_by_name["revision_1"]
    medium_daily = raw_by_name["revision_5"] / 5.0
    raw_by_name["revision_acceleration_1v5"] = short - medium_daily
    raw_by_name["negative_revision_acceleration_1v5"] = medium_daily - short

    median_level = np.zeros(len(days))
    for i in range(len(days)):
        values = consensus[i, np.isfinite(consensus[i])]
        if len(values) >= 3:
            median_level[i] = float(np.median(values))
    raw_by_name["level_reversion_1"] = -median_level * short

    for name, values in raw_by_name.items():
        names.append(name)
        rows.append(values)
    day_matrix = np.column_stack(rows).astype(np.float32)
    if not np.all(np.isfinite(day_matrix)):
        raise ValueError("non-finite revision feature")
    lookup = {day: day_matrix[i] for i, day in enumerate(days)}
    matrix = np.vstack([lookup[row[2]] for row in index]).astype(np.float32)
    return matrix, names, pd.DataFrame(availability)


def causality_check(index, references, sources, cutoff=dt.date(2025, 6, 30)):
    full, names, _ = build_revision_features(index, references, sources)
    changed = {}
    for source_name, local in sources.items():
        changed[source_name] = {}
        for code_number, (code, series) in enumerate(local.items(), start=1):
            values = series.values.copy()
            future = series.dates >= cutoff
            values[future] *= np.linspace(
                2.0 + code_number, 25.0 + 5.0 * code_number,
                int(future.sum()),
            ) ** code_number
            changed[source_name][code] = Series(code, series.dates.copy(), values)
    altered, altered_names, _ = build_revision_features(index, references, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future local-CB value changed a past revision feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future corruption did not affect future revision rows")
    return True


def _load_incumbent():
    with INCUMBENT_PATH.open("rb") as handle:
        return pickle.load(handle)[INCUMBENT]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    sources, digests = {}, {}
    for name, loader in SOURCES.items():
        sources[name], digests[name] = loader()
    matrix, names, availability = build_revision_features(index, references, sources)
    availability.to_csv(OUT / "availability_by_date.csv", index=False)
    causality_check(index, references, sources)

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
    finalists = {"incumbent": incumbent, "revision_selected": raw_outputs[selected_raw]}
    for weight in BLEND_WEIGHTS:
        name = f"incumbent{int((1-weight)*100)}_revision{int(weight*100)}"
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
        y5, dates, currencies, valid, masks, "crossbank_revision_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in finalists.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DC", "fixed_policy": POLICY,
        "sources": list(SOURCES), "payload_sha256": digests,
        "max_source_age_days": MAX_AGE_DAYS,
        "lags_in_signal_dates": LAGS,
        "raw_candidates": names, "blend_weights": BLEND_WEIGHTS,
        "selection_period": 2024, "raw_selected": selected_raw,
        "finalist_selected": selected_finalist,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "within_source_difference_before_aggregation": True,
        "minimum_paired_sources": 3,
        "asof_rule": "every local-CB date strictly before signal date; CBR USD/CNY date <= signal date",
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
