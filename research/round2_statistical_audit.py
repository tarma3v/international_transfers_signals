"""Dependence- and multiplicity-aware audit of round-two policies.

The four-week moving-block bootstrap preserves overlapping h=5 labels and the
same-day cross-currency dependence.  A circular date-shift negative control
keeps the temporal and cross-sectional structure of the target but destroys
alignment with signals; its max statistic is computed over every recorded
round-two architecture at its general-validation working point.
"""
from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import benefit_forward_only, build_targets
from research.extended_features import load_or_build

OUT = Path("results/research/round2")
SEED = 20260904
B = 4000
PERIODS = {"shock_2022_2023": (2022, 2023), "retrospective_2024_2026": (2024, 2025, 2026)}


def _benefit(series, index) -> np.ndarray:
    result = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, 5)
        if value is not None:
            result[row] = value
    return result


def _fired(outputs: dict, years: tuple[int, ...], dates: np.ndarray,
           currencies: np.ndarray, y: np.ndarray, rate: float,
           rolling_window: int | None, cooldown_days: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.zeros(len(y), dtype=bool)
    fired = np.zeros(len(y), dtype=bool)
    for year in years:
        if year not in outputs:
            continue
        z = outputs[year]
        ca = np.asarray(z["calib_idx"], dtype=int)
        te = np.asarray(z["test_idx"], dtype=int)
        valid[te] = ~np.isnan(y[te])
        for currency in CORRIDORS:
            cm = currencies[ca] == currency
            tm = currencies[te] == currency
            cal_order = np.argsort(dates[ca[cm]])
            test_order = np.argsort(dates[te[tm]])
            cal_scores = np.asarray(z["calib_score"])[cm][cal_order]
            test_rows = te[tm][test_order]
            test_scores = np.asarray(z["test_score"])[tm][test_order]
            if rolling_window:
                joined = np.concatenate([cal_scores, test_scores])
                cutoffs = (
                    pd.Series(joined).rolling(rolling_window, min_periods=1)
                    .quantile(1.0 - rate).shift(1).to_numpy()[len(cal_scores):]
                )
            else:
                cutoffs = np.full(len(test_rows), np.quantile(cal_scores, 1.0 - rate))
            last = None
            for row, score, cutoff in zip(test_rows, test_scores, cutoffs):
                enough_gap = last is None or (dates[row] - last).days >= cooldown_days
                if score >= cutoff and enough_gap:
                    fired[row] = True; last = dates[row]
    return valid, fired


def _bootstrap_all(y: np.ndarray, benefit: np.ndarray, dates: np.ndarray,
                   valid: np.ndarray, policies: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    frame = pd.DataFrame({"date": dates[valid], "y": y[valid], "benefit": benefit[valid]})
    frame["week"] = frame.date.map(lambda d: d - dt.timedelta(days=d.weekday()))
    for name, fired in policies.items():
        frame[name] = fired[valid]
    weekly = []
    for _week, z in frame.groupby("week", sort=True):
        row = [float(z.y.sum()), float(len(z))]
        for name in policies:
            active = z[name].to_numpy(bool)
            values = z.loc[active, "benefit"].dropna()
            row.extend([float(z.loc[active, "y"].sum()), float(active.sum()),
                        float(values.sum()), float(len(values))])
        weekly.append(row)
    stats = np.asarray(weekly, dtype=float)
    n_weeks, block = len(stats), 4
    n_blocks = int(np.ceil(n_weeks / block))
    rng = np.random.default_rng(SEED)
    result = {name: {"lift": np.empty(B), "benefit": np.empty(B)} for name in policies}
    for b in range(B):
        starts = rng.integers(0, n_weeks - block + 1, size=n_blocks)
        pick = np.concatenate([np.arange(s, s + block) for s in starts])[:n_weeks]
        totals = stats[pick].sum(axis=0)
        base = totals[0] / totals[1]
        for j, name in enumerate(policies):
            hit_y, n_fire, benefit_sum, benefit_n = totals[2 + 4*j: 6 + 4*j]
            result[name]["lift"][b] = (hit_y / n_fire) / base
            result[name]["benefit"][b] = benefit_sum / benefit_n
    return result


def _summary(name: str, y: np.ndarray, benefit: np.ndarray, dates: np.ndarray,
             valid: np.ndarray, fired: np.ndarray, draws: dict,
             anchor_draws: dict, anchor_lift: float) -> dict:
    active = valid & fired
    lift = float(y[active].mean() / y[valid].mean())
    b = benefit[active & ~np.isnan(benefit)]
    difference = draws["lift"] - anchor_draws["lift"]
    return {
        "policy": name, "n": int(active.sum()),
        "frequency": rate_per_week(int(active.sum()), len(CORRIDORS), dates, valid),
        "lift": lift,
        "forward_benefit_bps": float(np.mean(b)),
        "lift_ci_low": float(np.quantile(draws["lift"], .025)),
        "lift_ci_high": float(np.quantile(draws["lift"], .975)),
        "p_lift_le_1": float((np.sum(draws["lift"] <= 1.0) + 1) / (len(draws["lift"]) + 1)),
        "benefit_ci_low": float(np.quantile(draws["benefit"], .025)),
        "benefit_ci_high": float(np.quantile(draws["benefit"], .975)),
        "lift_diff_vs_anchor": float(lift - anchor_lift),
        "lift_diff_vs_anchor_ci_low": float(np.quantile(difference, .025)),
        "lift_diff_vs_anchor_ci_high": float(np.quantile(difference, .975)),
    }


def _load_finalists() -> dict[str, tuple[dict, float, int | None, int]]:
    with (Path("results/research") / "candidate_outputs_h5_v2.pkl").open("rb") as fh:
        old = pickle.load(fh)
    with (OUT / "diverse_outputs.pkl").open("rb") as fh: diverse = pickle.load(fh)
    with (OUT / "router_outputs.pkl").open("rb") as fh: router = pickle.load(fh)
    with (OUT / "recency_outputs.pkl").open("rb") as fh: recency = pickle.load(fh)
    with (OUT / "tower_outputs.pkl").open("rb") as fh: towers = pickle.load(fh)
    return {
        "anchor_multiscale_locked": (old["anchor_multiscale"], .20, 250, 0),
        "new_global_extra": (diverse["global_compact_extra"], .20, 250, 0),
        "router_equal": (router["equal"], .30, 250, 0),
        "router_regime_soft": (router["regime_soft"], .30, 500, 0),
        "router_global_soft": (router["global_soft"], .30, 250, 0),
        "recency_window_short": (recency["global_extra_window_short"], .20, None, 0),
        "recency_window3": (recency["global_extra_window3"], .30, None, 0),
        "local_to_global_xgb": (
            towers["local_logit_top80_5y__global_xgb_offset"], .30, 120, 0,
        ),
    }


def _all_recorded() -> dict[str, tuple[dict, float, int | None, int]]:
    result = {}
    families = (
        ("diverse", "diverse_outputs.pkl", "diverse_stage1.csv"),
        ("router", "router_outputs.pkl", "router_stage1.csv"),
        ("recency", "recency_outputs.pkl", "recency_stage1.csv"),
        ("ranker", "ranker_outputs.pkl", "ranker_stage1.csv"),
        ("tower", "tower_outputs.pkl", "tower_stage1_shortlist.csv"),
        ("external", "external_model_outputs.pkl", "external_stage1.csv"),
    )
    for family, outputs_file, selection_file in families:
        if not (OUT / outputs_file).exists() or not (OUT / selection_file).exists():
            continue
        with (OUT / outputs_file).open("rb") as fh: outputs = pickle.load(fh)
        selection = pd.read_csv(OUT / selection_file)
        for row in selection.itertuples(index=False):
            name = row.candidate
            if name not in outputs:
                continue
            rate = float(getattr(row, "rate_target", getattr(row, "stage1_rate_target", .3)))
            rolling = int(getattr(row, "rolling_window", 0)) or None
            cooldown = int(getattr(row, "cooldown_days", 0))
            result[f"{family}:{name}"] = (outputs[name], rate, rolling, cooldown)
    return result


def _circular_shift_audit(y: np.ndarray, dates: np.ndarray, currencies: np.ndarray,
                          valid: np.ndarray, policies: dict[str, np.ndarray],
                          period: str) -> pd.DataFrame:
    days = np.asarray(sorted(set(dates[valid])), dtype=object)
    row_lookup = {(dates[row], currencies[row]): row for row in np.where(valid)[0]}
    y_matrix = np.full((len(days), len(CORRIDORS)), np.nan)
    fire_matrices = {name: np.zeros_like(y_matrix, dtype=bool) for name in policies}
    for i, day in enumerate(days):
        for j, currency in enumerate(CORRIDORS):
            row = row_lookup.get((day, currency))
            if row is None:
                continue
            y_matrix[i, j] = y[row]
            for name, fired in policies.items(): fire_matrices[name][i, j] = fired[row]
    complete = np.all(np.isfinite(y_matrix), axis=1)
    y_matrix = y_matrix[complete]
    fire_matrices = {name: values[complete] for name, values in fire_matrices.items()}
    base = float(np.mean(y_matrix))
    observed = {name: float(np.mean(y_matrix[fire]) / base)
                for name, fire in fire_matrices.items() if fire.any()}
    allowed = np.asarray([k for k in range(20, len(y_matrix) - 20)])
    rng = np.random.default_rng(SEED)
    shifts = rng.choice(allowed, size=B, replace=True)
    null = {name: np.empty(B) for name in observed}
    max_null = np.empty(B)
    for b, shift in enumerate(shifts):
        shifted = np.roll(y_matrix, int(shift), axis=0)
        values = []
        for name in observed:
            value = float(np.mean(shifted[fire_matrices[name]]) / base)
            null[name][b] = value; values.append(value)
        max_null[b] = max(values)
    observed_max = max(observed.values())
    rows = []
    for name, value in observed.items():
        rows.append({
            "period": period, "policy": name, "observed_lift": value,
            "circular_shift_p_unadjusted": float((np.sum(null[name] >= value) + 1) / (B + 1)),
            "circular_shift_p_max_adjusted": float((np.sum(max_null >= value) + 1) / (B + 1)),
            "n_recorded_policies": len(observed), "observed_max_lift": observed_max,
            "null_max_lift_q95": float(np.quantile(max_null, .95)),
        })
    return pd.DataFrame(rows)


def _breakdown(policy: str, period: str, y: np.ndarray, benefit: np.ndarray,
               dates: np.ndarray, currencies: np.ndarray, valid: np.ndarray,
               fired: np.ndarray) -> list[dict]:
    rows = []
    groups = [("overall", "all", valid, len(CORRIDORS))]
    for year in sorted({d.year for d in dates[valid]}):
        mask = valid & np.asarray([d.year == year for d in dates])
        groups.append(("year", str(year), mask, len(CORRIDORS)))
    for currency in CORRIDORS:
        mask = valid & (currencies == currency)
        groups.append(("currency", currency, mask, 1))
    for kind, group, scope, n_corridors in groups:
        active = scope & fired
        b = benefit[active & ~np.isnan(benefit)]
        base = float(y[scope].mean())
        hit = float(y[active].mean()) if active.any() else np.nan
        rows.append({
            "period": period, "policy": policy, "breakdown": kind, "group": group,
            "n_scope": int(scope.sum()), "n_signals": int(active.sum()),
            "frequency": rate_per_week(int(active.sum()), n_corridors, dates, scope),
            "base_rate": base, "hit_rate": hit,
            "lift": hit / base if active.any() else np.nan,
            "forward_benefit_bps": float(np.mean(b)) if len(b) else np.nan,
        })
    return rows


def main() -> None:
    _X, _names, index, series = load_or_build()
    dates = np.asarray([day for _c, _i, day in index], dtype=object)
    currencies = np.asarray([currency for currency, _i, _d in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    benefit = _benefit(series, index)
    finalists = _load_finalists()
    all_recorded = _all_recorded()
    summary_rows, breakdown_rows, circular_frames = [], [], []
    for period, years in PERIODS.items():
        finalist_masks = {}
        common_valid = np.asarray([d.year in years for d in dates]) & ~np.isnan(y)
        for name, (outputs, rate, rolling, cooldown) in finalists.items():
            valid, fired = _fired(outputs, years, dates, currencies, y, rate, rolling, cooldown)
            if np.array_equal(valid, common_valid): finalist_masks[name] = fired
        draws = _bootstrap_all(y, benefit, dates, common_valid, finalist_masks)
        anchor_draws = draws["anchor_multiscale_locked"]
        anchor_fire = finalist_masks["anchor_multiscale_locked"] & common_valid
        anchor_lift = float(y[anchor_fire].mean() / y[common_valid].mean())
        for name, fired in finalist_masks.items():
            row = _summary(name, y, benefit, dates, common_valid, fired,
                           draws[name], anchor_draws, anchor_lift)
            row["period"] = period; summary_rows.append(row)
            breakdown_rows.extend(_breakdown(
                name, period, y, benefit, dates, currencies, common_valid, fired,
            ))

        recorded_masks = {}
        for name, (outputs, rate, rolling, cooldown) in all_recorded.items():
            valid, fired = _fired(outputs, years, dates, currencies, y, rate, rolling, cooldown)
            if np.array_equal(valid, common_valid) and fired.any(): recorded_masks[name] = fired
        circular_frames.append(_circular_shift_audit(
            y, dates, currencies, common_valid, recorded_masks, period,
        ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "round2_block_bootstrap.csv", index=False)
    pd.DataFrame(breakdown_rows).to_csv(OUT / "round2_finalist_breakdown.csv", index=False)
    circular = pd.concat(circular_frames, ignore_index=True)
    circular.to_csv(OUT / "round2_circular_shift_multiplicity.csv", index=False)
    print("\nFOUR-WEEK BLOCK BOOTSTRAP")
    print(summary.to_string(index=False))
    print("\nCIRCULAR-SHIFT MAX TEST: LOWEST ADJUSTED P")
    print(circular.sort_values(["period", "circular_shift_p_max_adjusted"])
          .groupby("period").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
