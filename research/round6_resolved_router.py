"""Packet-N weekly router driven by already resolved expert-signal outcomes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from ml.validation import target_reach_dates
from research.round2_statistical_audit import _bootstrap_all, _circular_shift_audit, _fired
from research.round5_features import load_round5_features


OUT = Path("results/research/round6/resolved_router")
EXPERTS = (
    "benefit_ranker_anchor25",
    "broad75_baseload25",
    "cbr_baseload",
)


@dataclass(frozen=True)
class RouterSpec:
    scope: str
    window_signals: int
    prior_strength: float


def specs() -> list[RouterSpec]:
    return [
        RouterSpec(scope, window, prior)
        for scope in ("global", "local", "hierarchical")
        for window in (10, 20, 40, 80)
        for prior in (5.0, 20.0, 50.0)
    ]


def load_experts():
    with Path("results/research/round6/direct_rankers/outputs.pkl").open("rb") as handle:
        rankers = pickle.load(handle)
    with Path("results/research/round6/broad_cbr_hybrid/outputs.pkl").open("rb") as handle:
        hybrids = pickle.load(handle)
    return {
        EXPERTS[0]: (
            rankers["rank_pair_benefit_compact_quarter_anchor25"], .22, 60,
        ),
        EXPERTS[1]: (hybrids["broad75_baseload25"], .35, 250),
        EXPERTS[2]: (rankers["packet_e_cbr_anchor50"], .35, 120),
    }


def _posterior(rows, y, prior_mean, prior_strength, window):
    if len(rows):
        rows = rows[-window:]
    return float(
        (float(np.sum(y[rows])) + prior_strength * prior_mean)
        / (len(rows) + prior_strength)
    )


def routed_fired(experts, target_years, dates, currencies, y, reach, spec):
    first_year, last_year = 2024, max(target_years)
    history_years = tuple(range(first_year, last_year + 1))
    masks = {}
    valid_all = None
    for name, (output, rate, rolling) in experts.items():
        valid, fired = _fired(
            output, history_years, dates, currencies, y, rate, rolling, 0,
        )
        if valid_all is None:
            valid_all = valid
        elif not np.array_equal(valid_all, valid):
            raise AssertionError(f"expert evaluation rows differ: {name}")
        masks[name] = fired

    target_valid = valid_all & np.asarray([day.year in target_years for day in dates])
    routed = np.zeros(len(y), dtype=bool)
    chosen = np.full(len(y), "", dtype=object)
    all_weeks = sorted({
        (day.isocalendar()[0], day.isocalendar()[1])
        for day in dates[valid_all]
    })
    for week_key in all_weeks:
        week_rows = np.where(valid_all & np.asarray([
            tuple(day.isocalendar()[:2]) == week_key for day in dates
        ]))[0]
        if not len(week_rows):
            continue
        week_start = min(dates[week_rows])
        resolved_global = np.asarray([
            np.isfinite(y[row]) and reach[row] <= week_start
            for row in range(len(y))
        ])
        prior_mean = float(np.mean(y[resolved_global])) if resolved_global.any() else .30
        global_rates = {}
        for name in EXPERTS:
            rows = np.where(resolved_global & masks[name])[0]
            rows = rows[np.argsort(np.asarray(reach, dtype=object)[rows], kind="stable")]
            global_rates[name] = _posterior(
                rows, y, prior_mean, spec.prior_strength, spec.window_signals,
            )
        for currency in CORRIDORS:
            current = week_rows[currencies[week_rows] == currency]
            if not len(current):
                continue
            reliability = []
            for name in EXPERTS:
                if spec.scope == "global":
                    value = global_rates[name]
                else:
                    local = np.where(
                        resolved_global & masks[name] & (currencies == currency)
                    )[0]
                    local = local[
                        np.argsort(np.asarray(reach, dtype=object)[local], kind="stable")
                    ]
                    local_prior = (
                        prior_mean if spec.scope == "local" else global_rates[name]
                    )
                    value = _posterior(
                        local, y, local_prior, spec.prior_strength,
                        spec.window_signals,
                    )
                reliability.append(value)
            winner = EXPERTS[int(np.argmax(reliability))]
            routed[current] = masks[winner][current]
            chosen[current] = winner
    return target_valid, routed, chosen


def summarize(experts, target_years, dates, currencies, y, benefit, reach, spec):
    valid, fired, chosen = routed_fired(
        experts, target_years, dates, currencies, y, reach, spec,
    )
    active = valid & fired
    base = float(y[valid].mean())
    gains = benefit[active & ~np.isnan(benefit)]
    corridor_lift, corridor_freq = [], []
    for currency in CORRIDORS:
        scope = valid & (currencies == currency)
        signals = active & (currencies == currency)
        corridor_lift.append(
            float(y[signals].mean() / y[scope].mean()) if signals.any() else np.nan
        )
        corridor_freq.append(rate_per_week(int(signals.sum()), 1, dates, scope))
    quarter_lift, quarter_freq, year_lift, year_freq = [], [], [], []
    for year in target_years:
        scope = valid & np.asarray([day.year == year for day in dates])
        signals = active & scope
        if scope.any() and signals.any():
            year_lift.append(float(y[signals].mean() / y[scope].mean()))
            year_freq.append(rate_per_week(int(signals.sum()), len(CORRIDORS), dates, scope))
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                day.year == year and (day.month - 1) // 3 + 1 == quarter
                for day in dates
            ])
            if not scope.any():
                continue
            signals = active & scope
            quarter_freq.append(rate_per_week(
                int(signals.sum()), len(CORRIDORS), dates, scope,
            ))
            quarter_lift.append(
                float(y[signals].mean() / y[scope].mean()) if signals.any() else np.nan
            )
    row = {
        "n": int(active.sum()),
        "frequency": rate_per_week(int(active.sum()), len(CORRIDORS), dates, valid),
        "hit_rate": float(y[active].mean()),
        "base_rate": base,
        "lift": float(y[active].mean() / base),
        "forward_benefit_bps": float(np.mean(gains)) if len(gains) else np.nan,
        "corridor_lift_min": float(np.nanmin(corridor_lift)),
        "corridor_freq_min": float(min(corridor_freq)),
        "corridor_freq_max": float(max(corridor_freq)),
        "quarter_lift_min": float(np.nanmin(quarter_lift)),
        "quarter_frequency_min": float(min(quarter_freq)),
        "quarter_frequency_max": float(max(quarter_freq)),
        "macro_year_lift": float(np.mean(year_lift)),
        "year_lift_min": float(min(year_lift)),
        "year_frequency_min": float(min(year_freq)),
        "year_frequency_max": float(max(year_freq)),
    }
    for name in EXPERTS:
        row[f"chosen_weeks_{name}"] = len({
            (dates[index].isocalendar()[0], dates[index].isocalendar()[1], currencies[index])
            for index in np.where(valid & (chosen == name))[0]
        })
    return row


def breakdown(experts, target_years, dates, currencies, y, benefit, reach, spec):
    valid, fired, chosen = routed_fired(
        experts, target_years, dates, currencies, y, reach, spec,
    )
    groups = [("overall", "all", valid, len(CORRIDORS))]
    for year in target_years:
        groups.append((
            "year", str(year), valid & np.asarray([d.year == year for d in dates]),
            len(CORRIDORS),
        ))
        for quarter in range(1, 5):
            scope = valid & np.asarray([
                d.year == year and (d.month - 1) // 3 + 1 == quarter for d in dates
            ])
            if scope.any():
                groups.append(("quarter", f"{year}Q{quarter}", scope, len(CORRIDORS)))
    for currency in CORRIDORS:
        groups.append(("currency", currency, valid & (currencies == currency), 1))
    rows = []
    for kind, group, scope, n_corridors in groups:
        active = scope & fired
        gains = benefit[active & ~np.isnan(benefit)]
        row = {
            "breakdown": kind, "group": group, "n_scope": int(scope.sum()),
            "n_signals": int(active.sum()),
            "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
            "base_rate": float(y[scope].mean()),
            "hit_rate": float(y[active].mean()) if active.any() else np.nan,
            "lift": float(y[active].mean() / y[scope].mean()) if active.any() else np.nan,
            "forward_benefit_bps": float(np.mean(gains)) if len(gains) else np.nan,
        }
        for name in EXPERTS:
            row[f"chosen_rows_{name}"] = int(np.sum(scope & (chosen == name)))
            row[f"signals_{name}"] = int(np.sum(active & (chosen == name)))
        rows.append(row)
    return rows


def unresolved_label_check(experts, dates, currencies, y, reach, spec):
    cut = dt.date(2025, 6, 30)
    valid, original, _chosen = routed_fired(
        experts, (2025,), dates, currencies, y, reach, spec,
    )
    changed_y = y.copy()
    future = np.asarray([
        np.isfinite(y[row]) and reach[row] > cut for row in range(len(y))
    ])
    changed_y[future] = 1.0 - changed_y[future]
    _valid, changed, _chosen = routed_fired(
        experts, (2025,), dates, currencies, changed_y, reach, spec,
    )
    past = valid & np.asarray([day <= cut for day in dates])
    if not np.array_equal(original[past], changed[past]):
        raise AssertionError("unresolved future label changed a past routed decision")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, _trajectory, _tnames, _paths = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    benefit = np.asarray([
        benefit_forward_only(series[currency].values, position, 5)
        if position + 5 < len(series[currency].values) else np.nan
        for currency, position, _day in index
    ])
    experts = load_experts()
    unresolved_label_check(experts, dates, currencies, y, reach, specs()[0])

    screen_rows = []
    for spec in specs():
        item = summarize(
            experts, (2024,), dates, currencies, y, benefit, reach, spec,
        )
        item.update(asdict(spec))
        screen_rows.append(item)
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(OUT / "screen_2024.csv", index=False)
    feasible = screen[
        screen.frequency.between(1.00, 2.00)
        & screen.corridor_freq_min.ge(.80)
        & screen.quarter_frequency_min.ge(.70)
        & screen.forward_benefit_bps.gt(0)
    ].copy()
    if feasible.empty:
        raise RuntimeError("no feasible packet-N router on 2024")
    feasible["robustness"] = feasible[["lift", "corridor_lift_min"]].min(axis=1)
    chosen = feasible.sort_values(
        ["robustness", "lift", "quarter_frequency_min"], ascending=False,
    ).iloc[0]
    chosen.to_frame().T.to_csv(OUT / "selected_2024.csv", index=False)
    spec = RouterSpec(
        str(chosen.scope), int(chosen.window_signals), float(chosen.prior_strength),
    )

    results_rows = []
    for period, target_years in (
        ("screen_2024", (2024,)),
        ("retrospective_2025", (2025,)),
        ("retrospective_2026", (2026,)),
        ("combined_2025_2026", (2025, 2026)),
    ):
        item = summarize(
            experts, target_years, dates, currencies, y, benefit, reach, spec,
        )
        item.update({"period": period, **asdict(spec)})
        results_rows.append(item)
    results = pd.DataFrame(results_rows)
    results.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(breakdown(
        experts, (2025, 2026), dates, currencies, y, benefit, reach, spec,
    )).to_csv(OUT / "breakdown_2025_2026.csv", index=False)

    valid, fired, _chosen = routed_fired(
        experts, (2025, 2026), dates, currencies, y, reach, spec,
    )
    masks = {"resolved_router": fired}
    draws = _bootstrap_all(y, benefit, dates, valid, masks)["resolved_router"]
    finite_lift = draws["lift"][np.isfinite(draws["lift"])]
    finite_benefit = draws["benefit"][np.isfinite(draws["benefit"])]
    pd.DataFrame([{
        "candidate": "resolved_router",
        "lift_ci_low": float(np.quantile(finite_lift, .025)),
        "lift_ci_high": float(np.quantile(finite_lift, .975)),
        "p_lift_le_1": float((np.sum(finite_lift <= 1) + 1) / (len(finite_lift) + 1)),
        "benefit_ci_low": float(np.quantile(finite_benefit, .025)),
        "benefit_ci_high": float(np.quantile(finite_benefit, .975)),
    }]).to_csv(OUT / "block_bootstrap.csv", index=False)
    _circular_shift_audit(
        y, dates, currencies, valid, masks, "retrospective_2025_2026",
    ).to_csv(OUT / "circular_shift_multiplicity.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "experts": {
            name: {"rate": rate, "rolling": rolling}
            for name, (_output, rate, rolling) in experts.items()
        },
        "router": asdict(spec),
        "selection_period": 2024,
        "history_carried_forward_from": 2024,
        "feedback_rule": "only expert signals with h5 reach date <= week start",
        "physical_unresolved_label_corruption_check": True,
        "next_rate_feature": False,
        "later_period_status": "protocol-controlled retrospective, not pristine",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[[
        "period", "frequency", "lift", "forward_benefit_bps",
        "corridor_freq_min", "corridor_lift_min", "quarter_frequency_min",
        "quarter_frequency_max", "quarter_lift_min",
        *[f"chosen_weeks_{name}" for name in EXPERTS],
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
