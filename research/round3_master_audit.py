"""Unified metric, stability, and uncertainty audit for all research rounds."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluate import rate_per_week
from ml.targets import build_targets
from research.extended_features import load_or_build
from research.model_study import evaluate
from research.round2_statistical_audit import (
    _bootstrap_all, _circular_shift_audit, _fired,
)
from research.round3_balanced_selection import _load as _load_balanced_pool
from research.round3_postshock_reset import _benefit

ROOT = Path("results/research")
R2 = ROOT / "round2"
OUT = ROOT / "round3"
PERIODS = {
    "general_2017_2020": (2017, 2018, 2019, 2020),
    "shock_2022_2023": (2022, 2023),
    "retrospective_2024_2026": (2024, 2025, 2026),
}


@dataclass
class Policy:
    name: str
    outputs: dict
    rate: float
    rolling: int | None
    cooldown: int = 0
    status: str = "strict past-only"


def _cache(path: Path) -> dict:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _policies() -> list[Policy]:
    old = _cache(ROOT / "candidate_outputs_h5_v2.pkl")
    diverse = _cache(R2 / "diverse_outputs.pkl")
    router = _cache(R2 / "router_outputs.pkl")
    recency = _cache(R2 / "recency_outputs.pkl")
    online = _cache(OUT / "online_mixture_outputs.pkl")
    reset = _cache(OUT / "postshock_reset_outputs.pkl")
    current = _cache(OUT / "current_ensemble_outputs.pkl")
    return [
        Policy("anchor_multiscale_locked", old["anchor_multiscale"], .20, 250,
               status="locked before final read"),
        Policy("anchor_trend_posthoc", old["anchor_trend"], .25, None,
               status="posthoc diagnostic"),
        Policy("global_extra", diverse["global_compact_extra"], .20, 250),
        Policy("router_equal_original", router["equal"], .30, 250),
        Policy("router_equal_balanced", router["equal"], .30, 500,
               status="policy selected with per-year frequency audit"),
        Policy("router_regime_soft", router["regime_soft"], .30, 500),
        Policy("recency_short_mix", recency["global_extra_window_short"], .20, None),
        Policy("round3_consensus_geometric", online["consensus_geometric"], .20, 120,
               status="round-3 finalist"),
        Policy("round3_online_local_headline", online["hedge_local_eta2_rho0p99"], .25, None,
               status="retrospective; unstable annual frequency"),
        Policy("postshock_reset_xgb_stable", reset["reset_xgb"], .20, 120,
               status="post-2022-only retrospective sensitivity"),
        Policy("recent_reset_anchor_blend",
               current["0.25*reset_xgb+0.75*multiscale_anchor"], .20, 120,
               status="2024-selected, 2025-gated retrospective blend"),
    ]


def _period_metric(policy: Policy, years: tuple[int, ...], y: np.ndarray,
                   symmetric: np.ndarray, forward: np.ndarray, dates: np.ndarray,
                   currencies: np.ndarray) -> dict | None:
    if not all(year in policy.outputs for year in years):
        return None
    metric = evaluate(
        policy.outputs, y, dates, currencies, forward, years,
        policy.rate, policy.rolling, policy.cooldown,
    )
    valid, fired = _fired(
        policy.outputs, years, dates, currencies, y,
        policy.rate, policy.rolling, policy.cooldown,
    )
    active = valid & fired
    annual = [evaluate(
        policy.outputs, y, dates, currencies, forward, (year,),
        policy.rate, policy.rolling, policy.cooldown,
    ) for year in years]
    metric.update({
        "policy": policy.name,
        "status": policy.status,
        "period": next(name for name, value in PERIODS.items() if value == years),
        "symmetric_benefit_bps": float(np.nanmean(symmetric[active])),
        "macro_year_lift": float(np.mean([row["lift"] for row in annual])),
        "year_frequency_min": float(np.min([row["frequency"] for row in annual])),
        "year_frequency_max": float(np.max([row["frequency"] for row in annual])),
    })
    metric["simpson_gap"] = metric["lift"] - metric["macro_year_lift"]
    return metric


def _breakdown(policy: Policy, y: np.ndarray, symmetric: np.ndarray,
               forward: np.ndarray, dates: np.ndarray,
               currencies: np.ndarray) -> list[dict]:
    years = PERIODS["retrospective_2024_2026"]
    if not all(year in policy.outputs for year in years):
        return []
    valid, fired = _fired(policy.outputs, years, dates, currencies, y,
                          policy.rate, policy.rolling, policy.cooldown)
    rows = []
    groups = []
    for year in years:
        groups.append(("year", str(year), valid & np.asarray([d.year == year for d in dates])))
    for currency in sorted(set(currencies)):
        groups.append(("currency", currency, valid & (currencies == currency)))
    for kind, group, scope in groups:
        active = scope & fired
        n_corridors = 5 if kind == "year" else 1
        rows.append({
            "policy": policy.name, "breakdown": kind, "group": group,
            "n_scope": int(scope.sum()), "n_signals": int(active.sum()),
            "base_rate": float(np.mean(y[scope])),
            "hit_rate": float(np.mean(y[active])) if active.any() else np.nan,
            "lift": float(np.mean(y[active]) / np.mean(y[scope])) if active.any() else np.nan,
            "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
            "forward_benefit_bps": float(np.nanmean(forward[active])) if active.any() else np.nan,
            "symmetric_benefit_bps": float(np.nanmean(symmetric[active])) if active.any() else np.nan,
        })
    return rows


def _bootstrap(policies: list[Policy], y: np.ndarray, symmetric: np.ndarray,
               forward: np.ndarray, dates: np.ndarray,
               currencies: np.ndarray) -> pd.DataFrame:
    years = PERIODS["retrospective_2024_2026"]
    common_valid = np.asarray([d.year in years for d in dates]) & ~np.isnan(y)
    masks = {}
    for policy in policies:
        if all(year in policy.outputs for year in years):
            valid, fired = _fired(policy.outputs, years, dates, currencies, y,
                                  policy.rate, policy.rolling, policy.cooldown)
            if np.array_equal(valid, common_valid):
                masks[policy.name] = fired
    forward_draws = _bootstrap_all(y, forward, dates, common_valid, masks)
    symmetric_draws = _bootstrap_all(y, symmetric, dates, common_valid, masks)
    anchor = forward_draws["anchor_multiscale_locked"]["lift"]
    rows = []
    for name, fired in masks.items():
        active = common_valid & fired
        lift = float(np.mean(y[active]) / np.mean(y[common_valid]))
        diff = forward_draws[name]["lift"] - anchor
        rows.append({
            "policy": name, "n": int(active.sum()), "lift": lift,
            "lift_ci_low": float(np.quantile(forward_draws[name]["lift"], .025)),
            "lift_ci_high": float(np.quantile(forward_draws[name]["lift"], .975)),
            "lift_diff_vs_locked_anchor": lift - float(
                np.mean(y[common_valid & masks["anchor_multiscale_locked"]])
                / np.mean(y[common_valid])
            ),
            "lift_diff_ci_low": float(np.quantile(diff, .025)),
            "lift_diff_ci_high": float(np.quantile(diff, .975)),
            "forward_benefit_bps": float(np.nanmean(forward[active])),
            "forward_ci_low": float(np.quantile(forward_draws[name]["benefit"], .025)),
            "forward_ci_high": float(np.quantile(forward_draws[name]["benefit"], .975)),
            "symmetric_benefit_bps": float(np.nanmean(symmetric[active])),
            "symmetric_ci_low": float(np.quantile(symmetric_draws[name]["benefit"], .025)),
            "symmetric_ci_high": float(np.quantile(symmetric_draws[name]["benefit"], .975)),
        })
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def _multiplicity(y: np.ndarray, dates: np.ndarray, currencies: np.ndarray) -> pd.DataFrame:
    years = PERIODS["retrospective_2024_2026"]
    common_valid = np.asarray([d.year in years for d in dates]) & ~np.isnan(y)
    stage1 = pd.read_csv(OUT / "balanced_stage1.csv").set_index("candidate")
    outputs = _load_balanced_pool()
    masks = {}
    for name, output in outputs.items():
        if name not in stage1.index:
            continue
        row = stage1.loc[name]
        valid, fired = _fired(output, years, dates, currencies, y,
                              float(row["rate"]), int(row["rolling"]) or None, 0)
        if np.array_equal(valid, common_valid) and fired.any():
            masks[name] = fired
    return _circular_shift_audit(
        y, dates, currencies, common_valid, masks,
        "retrospective_2024_2026_round3_pool",
    )


def main() -> None:
    _X, _names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y = targets["fav_h5"]
    symmetric = targets["benefit_h5"]
    forward = _benefit(series, index)
    policies = _policies()

    rows, breakdown = [], []
    for policy in policies:
        for years in PERIODS.values():
            row = _period_metric(policy, years, y, symmetric, forward, dates, currencies)
            if row is not None:
                rows.append(row)
        breakdown.extend(_breakdown(policy, y, symmetric, forward, dates, currencies))
    master = pd.DataFrame(rows)
    master.to_csv(OUT / "master_policy_metrics.csv", index=False)
    pd.DataFrame(breakdown).to_csv(OUT / "master_final_breakdown.csv", index=False)
    bootstrap = _bootstrap(policies, y, symmetric, forward, dates, currencies)
    bootstrap.to_csv(OUT / "round3_block_bootstrap.csv", index=False)
    circular = _multiplicity(y, dates, currencies)
    circular.to_csv(OUT / "round3_circular_shift_multiplicity.csv", index=False)

    columns = ["period", "policy", "frequency", "year_frequency_min",
               "year_frequency_max", "lift", "macro_year_lift", "simpson_gap",
               "forward_benefit_bps", "symmetric_benefit_bps", "year_lift_min",
               "corridor_lift_min"]
    print(master[columns].to_string(index=False))
    print("\nBOOTSTRAP", bootstrap.to_string(index=False), sep="\n")
    print("\nLOWEST MULTIPLICITY-ADJUSTED P")
    print(circular.sort_values("circular_shift_p_max_adjusted").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
