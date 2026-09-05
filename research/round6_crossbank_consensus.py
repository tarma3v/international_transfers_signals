"""Packet-CY: robust cross-bank shadow-RUB consensus.

Seven official local-central-bank archives each quote RUB, USD and CNY in a
different domestic currency.  The domestic unit cancels in USD/RUB and
CNY/RUB cross-rates, which gives seven comparable, strictly lagged estimates
of the common RUB market state.  This packet aggregates them without target
labels, selects one orientation/formula on 2024, and only then opens 2025-26.
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
from research.round6_armenian_central_bank_features import load_cba
from research.round6_belarus_nbrb_features import load_nbrb
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_georgia_nbg_features import load_nbg
from research.round6_kazakh_central_bank_features import load_kazakh_nbk
from research.round6_kyrgyz_central_bank_features import load_kyrgyz_nbkr
from research.round6_local_central_bank_features import load_nbt
from research.round6_multiobjective_blend import combine_causal
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_uzbek_central_bank_features import load_uzbek_cbu
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/crossbank_consensus")
INCUMBENT_PATH = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
INCUMBENT = "geometry75_cba_consensus_basis25"
MAX_AGE_DAYS = 7
SOURCES = {
    "armenia_cba": load_cba,
    "tajikistan_nbt": load_nbt,
    "uzbekistan_cbu": load_uzbek_cbu,
    "kazakhstan_nbk": load_kazakh_nbk,
    "kyrgyzstan_nbkr": load_kyrgyz_nbkr,
    "georgia_nbg": load_nbg,
    "belarus_nbrb": load_nbrb,
}
BASE_ORDER = (
    "median_consensus",
    "negative_median_consensus",
    "trimmed_consensus",
    "negative_trimmed_consensus",
    "fresh_weighted_consensus",
    "negative_fresh_weighted_consensus",
    "median_usd_basis",
    "negative_median_usd_basis",
    "median_cny_basis",
    "negative_median_cny_basis",
    "lower_quartile_consensus",
    "upper_quartile_consensus",
    "positive_breadth",
    "negative_dispersion",
    "signed_consensus_to_dispersion",
    "median_consensus_ex_cba",
)
DERIVED_LAGS = (1, 5, 20)
BLEND_WEIGHTS = (.05, .10, .20, .30)


def _last(series: Series, day: dt.date) -> tuple[float, int]:
    stop = int(np.searchsorted(series.dates, day, side="left"))
    if not stop:
        return np.nan, 999
    return float(series.values[stop - 1]), (day - series.dates[stop - 1]).days


def _cbr_last(series: Series, day: dt.date) -> float:
    stop = int(np.searchsorted(series.dates, day, side="right"))
    return float(series.values[stop - 1]) if stop else np.nan


def _source_bases(local: dict[str, Series], cbr_reference, day: dt.date):
    rub, rub_age = _last(local["RUB"], day)
    usd, usd_age = _last(local["USD"], day)
    cny, cny_age = _last(local["CNY"], day)
    age = max(rub_age, usd_age, cny_age)
    cbr_usd = _cbr_last(cbr_reference["USD"], day)
    cbr_cny = _cbr_last(cbr_reference["CNY"], day)
    available = (
        age <= MAX_AGE_DAYS
        and all(np.isfinite(x) and x > 0 for x in (rub, usd, cny, cbr_usd, cbr_cny))
    )
    if not available:
        return np.nan, np.nan, age
    usd_basis = float(np.log((usd / rub) / cbr_usd) * 10000.0)
    cny_basis = float(np.log((cny / rub) / cbr_cny) * 10000.0)
    return usd_basis, cny_basis, age


def build_crossbank_features(index, cbr_reference, sources):
    """Return label-free cross-source aggregates repeated on target rows."""
    unique_days = sorted({row[2] for row in index})
    by_day = {}
    diagnostics = []
    for day in unique_days:
        usd_values, cny_values, ages, names = [], [], [], []
        for name, local in sources.items():
            usd, cny, age = _source_bases(local, cbr_reference, day)
            if np.isfinite(usd) and np.isfinite(cny):
                usd_values.append(usd)
                cny_values.append(cny)
                ages.append(age)
                names.append(name)
        usd = np.asarray(usd_values, dtype=float)
        cny = np.asarray(cny_values, dtype=float)
        age = np.asarray(ages, dtype=float)
        consensus = .5 * (usd + cny)
        if len(consensus) < 3:
            values = np.zeros(len(BASE_ORDER), dtype=float)
        else:
            ordered = np.sort(consensus)
            trimmed = float(np.mean(ordered[1:-1])) if len(ordered) >= 5 else float(np.mean(ordered))
            weights = np.exp(-age / 3.0)
            weighted = float(np.average(consensus, weights=weights))
            median = float(np.median(consensus))
            dispersion = float(np.median(np.abs(consensus - median)))
            ex_cba = np.asarray([
                value for value, name in zip(consensus, names) if name != "armenia_cba"
            ], dtype=float)
            values = np.asarray([
                median,
                -median,
                trimmed,
                -trimmed,
                weighted,
                -weighted,
                float(np.median(usd)),
                -float(np.median(usd)),
                float(np.median(cny)),
                -float(np.median(cny)),
                float(np.quantile(consensus, .25)),
                float(np.quantile(consensus, .75)),
                float(np.mean(consensus > 0.0)),
                -dispersion,
                float(median / max(dispersion, 25.0)),
                float(np.median(ex_cba)) if len(ex_cba) else 0.0,
            ], dtype=float)
        by_day[day] = values
        diagnostics.append({
            "date": day,
            "source_count": len(consensus),
            "source_names": ",".join(names),
        })
    matrix = np.vstack([by_day[row[2]] for row in index]).astype(np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite cross-bank feature")
    return matrix, list(BASE_ORDER), pd.DataFrame(diagnostics)


def feature_family(matrix, names, index):
    result = {name: matrix[:, i].astype(float) for i, name in enumerate(names)}
    base = result["median_consensus"]
    for lag in DERIVED_LAGS:
        stale = delayed_by_currency(base[:, None], index, rows=lag)[:, 0]
        result[f"median_consensus_change_{lag}"] = base - stale
        result[f"median_consensus_reversal_{lag}"] = stale - base
    return result


def causality_check(index, cbr_reference, sources, cutoff=dt.date(2025, 6, 30)):
    full, names, _ = build_crossbank_features(index, cbr_reference, sources)
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
    altered, altered_names, _ = build_crossbank_features(index, cbr_reference, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future local-CB value changed a past cross-bank feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future corruption did not affect future cross-bank rows")
    return True


def _load_output(path: Path, name: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[name]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    sources, digests = {}, {}
    for name, loader in SOURCES.items():
        source, digest = loader()
        sources[name] = source
        digests[name] = digest
    matrix, names, availability = build_crossbank_features(index, references, sources)
    availability.to_csv(OUT / "availability_by_date.csv", index=False)
    causality_check(index, references, sources)
    raw = feature_family(matrix, names, index)

    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    y5 = targets["fav_h5"]
    raw_outputs = {name: _outputs(score, y5, dates) for name, score in raw.items()}
    raw_screen = horizon_rows(raw_outputs, (2024,), targets, forwards, dates, currencies)
    raw_summary = summarize(raw_screen)
    selected_raw = _choose(raw_summary)

    incumbent = _load_output(INCUMBENT_PATH, INCUMBENT)
    finalists = {"incumbent": incumbent, "crossbank_selected": raw_outputs[selected_raw]}
    for weight in BLEND_WEIGHTS:
        name = f"incumbent{int((1-weight)*100)}_crossbank{int(weight*100)}"
        finalists[name] = combine_causal(
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

    h5_rows = []
    for candidate, output in finalists.items():
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
        pickle.dump(finalists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bootstrap, masks, valid = _bootstrap(
        h5[h5.period == "screen_2024"], finalists, (2025, 2026),
        y5, forwards[5], dates, currencies,
    )
    bootstrap.to_csv(OUT / "block_bootstrap_h5.csv", index=False)
    _circular_shift_audit(
        y5, dates, currencies, valid, masks, "crossbank_consensus_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in finalists.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CY", "fixed_policy": POLICY,
        "sources": list(SOURCES), "payload_sha256": digests,
        "max_source_age_days": MAX_AGE_DAYS,
        "raw_candidates": list(raw), "blend_weights": BLEND_WEIGHTS,
        "selection_period": 2024, "raw_selected": selected_raw,
        "finalist_selected": selected_finalist,
        "selection_objective": "maximum worst official case-lift over h=1/3/5/10/20 with positive benefits",
        "asof_rule": "every local-CB date strictly before signal date; CBR USD/CNY date <= signal date",
        "domestic_currency_cancels": "local USD/local RUB and local CNY/local RUB",
        "label_free_aggregation": True,
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
