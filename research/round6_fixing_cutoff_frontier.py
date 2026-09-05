"""Packet-EI: screen a fixed frontier of pre-publication CNY fixing cutoffs."""
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
from research.round6_fixing_proxies import proxy_scores
from research.round6_moex_spot_1530_features import (
    SESSION_START,
    _arrays,
    _ratio,
    _reference_last,
    load_spot_1530_history,
)
from research.round6_resolved_models import _bootstrap, _breakdown, _evaluate
from research.round6_state_agreement_geometry import future_rank_check
from research.round6_uzbek_central_bank_models import (
    _forward,
    horizon_rows,
    summarize,
)


OUT = Path("results/research/round6/fixing_cutoff_frontier")
NOON_PATH = Path("results/research/round6/three_view_futures_consensus/outputs.pkl")
CUTOFFS = (
    dt.time(10, 30), dt.time(11, 30), dt.time(12, 30), dt.time(13, 30),
    dt.time(14, 30), dt.time(15, 0), dt.time(15, 20), dt.time(15, 30),
)
STALE_ROWS = 20


def _load(path: Path, candidate: str):
    with path.open("rb") as handle:
        return pickle.load(handle)[candidate]


def cutoff_name(cutoff):
    return f"cutoff_{cutoff.hour:02d}{cutoff.minute:02d}"


def cutoff_scores(index, history, references, cutoff):
    item = _arrays(history)["CNYRUB_TOM"]
    result = np.zeros(len(index), dtype=np.float32)
    for row_number, (_currency, _position, day) in enumerate(index):
        start_time = dt.datetime.combine(day, SESSION_START)
        cutoff_time = dt.datetime.combine(day, cutoff)
        start = int(np.searchsorted(item["begin"], start_time, side="left"))
        stop = int(np.searchsorted(item["end"], cutoff_time, side="left"))
        rows = np.arange(start, stop, dtype=int)
        rows = rows[item["begin"][rows] >= start_time]
        reference = _reference_last(references["CNY"], day)
        if len(rows) and np.isfinite(reference):
            result[row_number] = _ratio(float(np.mean(item["close"][rows])), reference)
    return result


def cutoff_causality_check(
    index, history, references, cutoff, boundary=dt.date(2025, 6, 30),
):
    original = cutoff_scores(index, history, references, cutoff)
    changed = {}
    boundary_time = dt.datetime.combine(boundary, cutoff)
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["end"] >= boundary_time:
                for key in ("open", "close", "high", "low"):
                    clone[key] *= 100.0
            changed[ticker].append(clone)
    altered = cutoff_scores(index, changed, references, cutoff)
    past = np.asarray([row[2] <= boundary for row in index])
    np.testing.assert_array_equal(original[past], altered[past])
    if not np.any(original[~past] != altered[~past]):
        raise AssertionError(f"future corruption inert at cutoff {cutoff}")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y5 = targets["fav_h5"]
    forwards = {h: _forward(series, index, h) for h in HORIZONS}

    raw = {}
    for cutoff in CUTOFFS:
        cutoff_causality_check(index, history, references, cutoff)
        score = cutoff_scores(index, history, references, cutoff)
        future_rank_check(score, dates, currencies)
        raw[cutoff_name(cutoff)] = score
    np.testing.assert_array_equal(
        raw["cutoff_1530"], proxy_scores(index, history, references)[:, 0],
    )
    outputs = {
        name: _outputs(score, y5, dates) for name, score in raw.items()
    }
    screen = horizon_rows(
        outputs, (2024,), targets, forwards, dates, currencies,
    )
    screen_summary = summarize(screen)
    screen_h5 = []
    for name, output in outputs.items():
        item = _evaluate(
            output, (2024,), POLICY, y5, forwards[5], dates, currencies,
        )
        screen_h5.append({"candidate": name, "frequency": item["frequency"]})
    screen_summary = screen_summary.merge(pd.DataFrame(screen_h5), on="candidate")
    feasible = screen_summary[
        screen_summary.frequency.between(1.0, 2.0)
        & screen_summary.symmetric_benefit_min.gt(0)
        & screen_summary.future_benefit_min.gt(0)
    ]
    if feasible.empty:
        raise RuntimeError("no feasible cutoff on 2024")
    selected = str(feasible.sort_values(
        ["horizon_lift_min", "horizon_lift_mean"], ascending=False,
    ).iloc[0].candidate)
    selected_stale_score = delayed_by_currency(
        raw[selected][:, None], index, rows=STALE_ROWS,
    )[:, 0]
    comparison = {
        "noon_consensus": _load(NOON_PATH, "selected"),
        "cutoff_1530": outputs["cutoff_1530"],
        "selected": outputs[selected],
        "matched_stale20": _outputs(selected_stale_score, y5, dates),
    }
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
        y5, dates, currencies, valid, masks, "fixing_cutoff_frontier_h5",
    ).to_csv(OUT / "circular_shift_h5.csv", index=False)
    breakdown = []
    for candidate, output in comparison.items():
        breakdown.extend(_breakdown(
            candidate, output, (2025, 2026), POLICY,
            y5, forwards[5], dates, currencies,
        ))
    pd.DataFrame(breakdown).to_csv(OUT / "breakdown_h5.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "EI",
        "cutoffs": [cutoff.isoformat() for cutoff in CUTOFFS],
        "score": "positive CNY completed-candle mean basis to current CBR",
        "fixed_policy": POLICY,
        "selection_period": 2024,
        "selected": selected,
        "payload_sha256": digest,
        "strict_asof": "candle end < signal-date cutoff",
        "all_cutoffs_physical_future_corruption_check": True,
        "cutoff_1530_exactly_matches_packet_eb": True,
        "missing_market_day_score": 0.0,
        "stale_rows": STALE_ROWS,
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
