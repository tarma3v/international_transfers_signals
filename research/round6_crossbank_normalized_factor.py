"""Packet-DD: causally normalized latent cross-bank RUB factor."""
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
from research.round6_broad_cbr_features import load_broad_features
from research.round6_crossbank_consensus import (
    INCUMBENT,
    INCUMBENT_PATH,
    MAX_AGE_DAYS,
    SOURCES,
    _source_bases,
)
from research.round6_cny_decomposition import POLICY
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/crossbank_normalized_factor")
WINDOW = 250
MIN_HISTORY = 60
BLEND_WEIGHTS = (.05, .10, .20, .30)
FEATURE_NAMES = (
    "median_percentile",
    "negative_median_percentile",
    "mean_percentile",
    "negative_mean_percentile",
    "lower_quartile_percentile",
    "upper_quartile_percentile",
    "low_extreme_breadth",
    "high_extreme_breadth",
    "median_robust_z",
    "negative_median_robust_z",
    "trimmed_robust_z",
    "negative_trimmed_robust_z",
    "negative_z_dispersion",
    "signed_z_to_dispersion",
    "cba_high_agreement",
    "cba_low_agreement",
)


def _causal_normalize(values):
    ranks = np.full_like(values, np.nan, dtype=float)
    zscores = np.full_like(values, np.nan, dtype=float)
    for j in range(values.shape[1]):
        history = []
        for i, value in enumerate(values[:, j]):
            if np.isfinite(value) and len(history) >= MIN_HISTORY:
                reference = np.asarray(history[-WINDOW:], dtype=float)
                ranks[i, j] = float(
                    np.searchsorted(np.sort(reference), value, side="right")
                    / len(reference)
                )
                center = float(np.median(reference))
                mad = float(np.median(np.abs(reference - center)))
                zscores[i, j] = (value - center) / max(1.4826 * mad, 25.0)
            if np.isfinite(value):
                history.append(float(value))
    return ranks, zscores


def build_normalized_features(index, references, sources):
    days = sorted({row[2] for row in index})
    source_names = tuple(sources)
    bases = np.full((len(days), len(source_names)), np.nan)
    availability = []
    for i, day in enumerate(days):
        present = []
        for j, name in enumerate(source_names):
            usd, cny, _age = _source_bases(sources[name], references, day)
            if np.isfinite(usd) and np.isfinite(cny):
                bases[i, j] = .5 * (usd + cny)
                present.append(name)
        availability.append({
            "date": day,
            "source_count": len(present),
            "source_names": ",".join(present),
        })
    ranks, zscores = _causal_normalize(bases)
    features = np.zeros((len(days), len(FEATURE_NAMES)), dtype=float)
    cba_index = source_names.index("armenia_cba")
    for i in range(len(days)):
        valid = np.isfinite(ranks[i]) & np.isfinite(zscores[i])
        if int(valid.sum()) < 3:
            continue
        rank = ranks[i, valid]
        zscore = zscores[i, valid]
        ordered = np.sort(zscore)
        trimmed = float(np.mean(ordered[1:-1])) if len(ordered) >= 5 else float(np.mean(ordered))
        median_rank = float(np.median(rank))
        mean_rank = float(np.mean(rank))
        median_z = float(np.median(zscore))
        z_dispersion = float(np.median(np.abs(zscore - median_z)))
        cba_rank = ranks[i, cba_index]
        cba_high = (
            min(float(cba_rank), median_rank) if np.isfinite(cba_rank) else 0.0
        )
        cba_low = (
            min(1.0 - float(cba_rank), 1.0 - median_rank)
            if np.isfinite(cba_rank) else 0.0
        )
        features[i] = (
            median_rank,
            -median_rank,
            mean_rank,
            -mean_rank,
            float(np.quantile(rank, .25)),
            float(np.quantile(rank, .75)),
            float(np.mean(rank < .25)),
            float(np.mean(rank > .75)),
            median_z,
            -median_z,
            trimmed,
            -trimmed,
            -z_dispersion,
            median_z / max(z_dispersion, .25),
            cba_high,
            cba_low,
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("non-finite normalized cross-bank feature")
    lookup = {day: features[i] for i, day in enumerate(days)}
    matrix = np.vstack([lookup[row[2]] for row in index]).astype(np.float32)
    return matrix, list(FEATURE_NAMES), pd.DataFrame(availability)


def causality_check(index, references, sources, cutoff=dt.date(2025, 6, 30)):
    full, names, _ = build_normalized_features(index, references, sources)
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
    altered, altered_names, _ = build_normalized_features(index, references, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future value changed a past normalized factor")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future corruption did not affect future normalized rows")
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
    matrix, names, availability = build_normalized_features(index, references, sources)
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
    finalists = {"incumbent": incumbent, "factor_selected": raw_outputs[selected_raw]}
    for weight in BLEND_WEIGHTS:
        name = f"incumbent{int((1-weight)*100)}_factor{int(weight*100)}"
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
        y5, dates, currencies, valid, masks, "crossbank_normalized_factor_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in finalists.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "DD", "fixed_policy": POLICY,
        "sources": list(SOURCES), "payload_sha256": digests,
        "max_source_age_days": MAX_AGE_DAYS,
        "normalization_window_signal_dates": WINDOW,
        "minimum_prior_observations": MIN_HISTORY,
        "raw_candidates": names, "blend_weights": BLEND_WEIGHTS,
        "selection_period": 2024, "raw_selected": selected_raw,
        "finalist_selected": selected_finalist,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "score_before_history_update": True,
        "minimum_current_sources": 3,
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
