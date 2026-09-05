"""Packet-CJ: block-bootstrap uncertainty for every required case horizon."""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import HORIZONS, benefit_forward_only, build_targets
from research.round2_statistical_audit import B, SEED
from research.round5_features import load_round5_features
from research.round6_cny_decomposition import POLICY
from research.round6_multihorizon_case_audit import corridor_period_adjusted_lift
from research.round6_resolved_models import _fire


OUT = Path("results/research/round6/multihorizon_uncertainty")
SOURCE = Path("results/research/round6/armenian_central_bank_models/outputs.pkl")
CANDIDATE = "geometry75_cba_consensus_basis25"
PERIODS = {
    "retrospective_2025": (2025,),
    "retrospective_2026": (2026,),
    "combined_2025_2026": (2025, 2026),
}


def _load():
    with SOURCE.open("rb") as handle:
        return pickle.load(handle)[CANDIDATE]


def _forward(series, index, h):
    result = np.full(len(index), np.nan)
    for row, (currency, position, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, position, h)
        if value is not None:
            result[row] = value
    return result


def _weekly_stats(y, symmetric, forward, valid, fired, dates, currencies):
    years = np.asarray([day.year for day in dates])
    weeks = np.asarray([day - dt.timedelta(days=day.weekday()) for day in dates], dtype=object)
    unique_weeks = sorted(set(weeks[valid]))
    group_keys = sorted(set(zip(currencies[valid], years[valid])))
    stats = np.zeros((len(unique_weeks), len(group_keys), 8), dtype=float)
    for wi, week in enumerate(unique_weeks):
        for gi, (currency, year) in enumerate(group_keys):
            scope = valid & (weeks == week) & (currencies == currency) & (years == year)
            active = scope & fired
            sym = active & np.isfinite(symmetric)
            fwd = active & np.isfinite(forward)
            stats[wi, gi] = (
                float(np.nansum(y[scope])), float(scope.sum()),
                float(np.nansum(y[active])), float(active.sum()),
                float(np.nansum(symmetric[sym])), float(sym.sum()),
                float(np.nansum(forward[fwd])), float(fwd.sum()),
            )
    return stats


def _bootstrap(stats, seed):
    n_weeks, _groups, _columns = stats.shape
    block = 4
    n_blocks = int(np.ceil(n_weeks / block))
    rng = np.random.default_rng(seed)
    lift = np.full(B, np.nan)
    symmetric = np.full(B, np.nan)
    forward = np.full(B, np.nan)
    for draw in range(B):
        starts = rng.integers(0, n_weeks - block + 1, size=n_blocks)
        picked = np.concatenate([np.arange(start, start + block) for start in starts])[:n_weeks]
        totals = stats[picked].sum(axis=0)
        expected_hits = 0.0
        for group in totals:
            total_y, total_n, _fire_y, fire_n = group[:4]
            if total_n > 0:
                expected_hits += fire_n * total_y / total_n
        fire_y = totals[:, 2].sum()
        if expected_hits > 0:
            lift[draw] = fire_y / expected_hits
        sym_n = totals[:, 5].sum()
        fwd_n = totals[:, 7].sum()
        if sym_n > 0:
            symmetric[draw] = totals[:, 4].sum() / sym_n
        if fwd_n > 0:
            forward[draw] = totals[:, 6].sum() / fwd_n
    return lift, symmetric, forward


def _holm(values):
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[position])))
        result[position] = running
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_rest = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    output = _load()
    rows = []
    family_size = len(HORIZONS) * len(PERIODS)
    family_lower_q = .05 / family_size
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        symmetric = targets[f"benefit_h{h}"]
        forward = _forward(series, index, h)
        for period_index, (period, years) in enumerate(PERIODS.items()):
            valid, fired = _fire(output, years, POLICY, y, dates, currencies)
            active = valid & fired
            observed_lift, _base, _macro = corridor_period_adjusted_lift(
                y, valid, fired, currencies, dates, years,
            )
            stats = _weekly_stats(y, symmetric, forward, valid, fired, dates, currencies)
            lift_draw, sym_draw, fwd_draw = _bootstrap(
                stats, SEED + 100 * h + period_index,
            )
            rows.append({
                "candidate": CANDIDATE, "period": period, "horizon": h,
                "n_scope": int(valid.sum()), "n_signals": int(active.sum()),
                "case_lift": observed_lift,
                "lift_ci_low": float(np.nanquantile(lift_draw, .025)),
                "lift_ci_high": float(np.nanquantile(lift_draw, .975)),
                "lift_familywise_lower": float(np.nanquantile(lift_draw, family_lower_q)),
                "p_lift_le_1p30": float((np.sum(lift_draw <= 1.30) + 1) / (len(lift_draw) + 1)),
                "symmetric_benefit_bps": float(np.nanmean(symmetric[active])),
                "symmetric_ci_low": float(np.nanquantile(sym_draw, .025)),
                "symmetric_ci_high": float(np.nanquantile(sym_draw, .975)),
                "symmetric_familywise_lower": float(np.nanquantile(sym_draw, family_lower_q)),
                "p_symmetric_le_zero": float((np.sum(sym_draw <= 0) + 1) / (len(sym_draw) + 1)),
                "future_only_benefit_bps": float(np.nanmean(forward[active])),
                "future_ci_low": float(np.nanquantile(fwd_draw, .025)),
                "future_ci_high": float(np.nanquantile(fwd_draw, .975)),
                "future_familywise_lower": float(np.nanquantile(fwd_draw, family_lower_q)),
                "p_future_le_zero": float((np.sum(fwd_draw <= 0) + 1) / (len(fwd_draw) + 1)),
            })
    results = pd.DataFrame(rows)
    results["p_lift_holm_15"] = _holm(results.p_lift_le_1p30.to_numpy())
    results["p_symmetric_holm_15"] = _holm(results.p_symmetric_le_zero.to_numpy())
    results["p_future_holm_15"] = _holm(results.p_future_le_zero.to_numpy())
    results["lift_1p30_supported_holm"] = results.p_lift_holm_15 < .05
    results["symmetric_positive_supported_holm"] = results.p_symmetric_holm_15 < .05
    results["future_positive_supported_holm"] = results.p_future_holm_15 < .05
    results.to_csv(OUT / "bootstrap_all_horizons.csv", index=False)
    gates = {
        "all_15_point_lifts_above_1p30": bool(results.case_lift.ge(1.30).all()),
        "all_15_point_symmetric_benefits_positive": bool(
            results.symmetric_benefit_bps.gt(0).all()
        ),
        "all_15_point_future_benefits_positive": bool(
            results.future_only_benefit_bps.gt(0).all()
        ),
        "all_15_lifts_supported_after_holm": bool(
            results.lift_1p30_supported_holm.all()
        ),
        "all_15_symmetric_benefits_supported_after_holm": bool(
            results.symmetric_positive_supported_holm.all()
        ),
        "all_15_future_benefits_supported_after_holm": bool(
            results.future_positive_supported_holm.all()
        ),
    }
    (OUT / "protocol.json").write_text(json.dumps({
        "packet": "CJ", "candidate": CANDIDATE, "fixed_policy": POLICY,
        "horizons": HORIZONS, "periods": PERIODS, "bootstrap_draws": B,
        "bootstrap": "moving four-week blocks; all currencies kept within sampled weeks",
        "case_lift": "corridor-year matched random-day expectation",
        "multiplicity": "Holm separately across 15 lift, symmetric-benefit, and future-benefit tests",
        "familywise_lower_quantile": family_lower_q,
        "gates": gates, "model_refit": False, "next_cbr_rate_used": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "period", "horizon", "case_lift", "lift_ci_low", "lift_ci_high",
        "p_lift_holm_15", "symmetric_benefit_bps", "symmetric_ci_low",
        "p_symmetric_holm_15", "future_only_benefit_bps", "future_ci_low",
        "p_future_holm_15",
    ]].sort_values(["period", "horizon"]).to_string(index=False))
    print("\nGATES\n" + json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
