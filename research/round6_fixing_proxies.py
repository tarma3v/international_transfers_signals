"""Packet-EB: unweighted candle-level proxies for the 15:30 CNY fixing."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from research.round2_statistical_audit import _circular_shift_audit
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_moex_spot_1530_features import (
    DECISION_TIME,
    SESSION_START,
    _arrays,
    _ratio,
    _reference_last,
    build_spot_1530_features,
    causality_check as source_causality_check,
    load_spot_1530_history,
)
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_state_agreement_geometry import future_rank_check
from research.round6_uzbek_central_bank_models import (
    _choose,
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/fixing_proxies")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
PROXIES = (
    "mean_close",
    "geometric_mean_close",
    "mean_ohlc4",
    "mean_hlc3",
    "mean_midpoint",
)
STALE_ROWS = 20


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def proxy_scores(index, history, references):
    item = _arrays(history)["CNYRUB_TOM"]
    # Missing market sessions remain eligible target days with a neutral score,
    # exactly as in the frozen packet-DY matrix.  NaN here would silently remove
    # weekends/holidays from both the policy and its base-rate denominator.
    result = np.zeros((len(index), len(PROXIES)), dtype=np.float32)
    for row_number, (_currency, _position, day) in enumerate(index):
        start_time = dt.datetime.combine(day, SESSION_START)
        cutoff = dt.datetime.combine(day, DECISION_TIME)
        start = int(np.searchsorted(item["begin"], start_time, side="left"))
        stop = int(np.searchsorted(item["end"], cutoff, side="left"))
        rows = np.arange(start, stop, dtype=int)
        rows = rows[item["begin"][rows] >= start_time]
        reference = _reference_last(references["CNY"], day)
        if not len(rows) or not np.isfinite(reference):
            continue
        opens = item["open"][rows]
        highs = item["high"][rows]
        lows = item["low"][rows]
        closes = item["close"][rows]
        levels = (
            float(np.mean(closes)),
            float(np.exp(np.mean(np.log(closes)))),
            float(np.mean((opens + highs + lows + closes) / 4.0)),
            float(np.mean((highs + lows + closes) / 3.0)),
            float(np.mean((highs + lows) / 2.0)),
        )
        result[row_number] = [_ratio(level, reference) for level in levels]
    return result


def proxy_causality_check(index, history, references, cutoff=dt.date(2025, 6, 30)):
    original = proxy_scores(index, history, references)
    changed = {}
    cutoff_time = dt.datetime.combine(cutoff, DECISION_TIME)
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["end"] >= cutoff_time:
                for key in ("open", "close", "high", "low"):
                    clone[key] *= 100.0
            changed[ticker].append(clone)
    altered = proxy_scores(index, changed, references)
    past = np.asarray([row[2] <= cutoff for row in index])
    np.testing.assert_array_equal(original[past], altered[past])
    if not np.any(original[~past] != altered[~past]):
        raise AssertionError("future corruption did not affect fixing proxies")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    source_causality_check(index, history, references)
    matrix = proxy_scores(index, history, references)
    proxy_causality_check(index, history, references)
    frozen_matrix, frozen_names = build_spot_1530_features(
        index, history, references,
    )
    frozen_mean = frozen_matrix[:, frozen_names.index(
        "moex_1530_cnyrub_tom_mean_cbr_basis"
    )]
    np.testing.assert_array_equal(matrix[:, 0], frozen_mean)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    noon = _load(NOON_PATH, "selected")
    candidates = {"noon_consensus": noon}
    stale_outputs = {}
    for column, name in enumerate(PROXIES):
        score = matrix[:, column]
        future_rank_check(score, dates, currencies)
        candidates[name] = _outputs(score, y5, dates)
        stale_score = delayed_by_currency(
            score[:, None], index, rows=STALE_ROWS,
        )[:, 0]
        stale_outputs[name] = _outputs(stale_score, y5, dates)

    screen = horizon_rows(
        candidates, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    selected = _choose(screen_summary)
    comparison = {"noon_consensus": noon, "selected": candidates[selected]}
    if selected in stale_outputs:
        comparison["matched_stale20"] = stale_outputs[selected]
    screen.to_csv(OUT / "screen_2024_by_horizon.csv", index=False)
    screen_summary.to_csv(OUT / "screen_2024_summary.csv", index=False)

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
        y5, dates, currencies, valid, masks, "fixing_proxies_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EB", "fixed_policy": POLICY,
        "payload_sha256": digest,
        "decision_time": "15:30:00 Europe/Moscow",
        "strict_asof": "10-minute candle end < signal date 15:30",
        "proxies": PROXIES,
        "proxy_sign": "positive proxy-current-CBR basis is favourable",
        "reported_value_rows": 0,
        "reported_volume_rows": 0,
        "true_vwap_claimed": False,
        "stale_rows": STALE_ROWS,
        "selection_period": 2024,
        "selected": selected,
        "next_cbr_rate_used": False,
        "later_period_status": (
            "protocol-controlled retrospective opened after 2024 selection"
        ),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Selected on 2024:", selected)
    print("\nSCREEN\n" + screen_summary.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).to_string(index=False))
    print("\nLATER\n" + later_summary.to_string(index=False))
    print("\nLATER BY HORIZON\n" + later[
        later.period == "combined_2025_2026"
    ].to_string(index=False))


if __name__ == "__main__":
    main()
