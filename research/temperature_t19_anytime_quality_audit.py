"""T19: unified clock/currency/year audit of the frozen anytime widget."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS, benefit_forward_only, target_now_favourable
from ml.transfer_temperature import (
    MOSCOW,
    RECEIPT_DEPENDENT_SOURCE_KINDS,
)


OUT = Path("results/research/temperature/t19_anytime_quality_audit")
BASE = Path("results/research/temperature/t17_spot_availability_repair")
PANEL = Path("results/research/after_publication/ap51_benefit/announcement_panel.csv")
REGISTERED = Path("research/temperature_t19_anytime_quality_audit_registered.md")
CLOCKS = (
    "00:15", "06:00", "09:15", "10:15", "10:45", "11:45", "12:45",
    "13:45", "14:45", "15:10", "15:25", "15:45", "16:45", "17:45",
    "18:45", "19:15", "20:15", "21:15", "22:15", "23:15",
)
SCENARIOS = ("calendar_assumed_replay", "no_same_day_receipt")
BLOCKS = (20, 50)
BOOTSTRAP_DRAWS = 1000
EMBARGO_DAYS = 2


def _ece(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return np.nan
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, 9)
    return float(sum(
        np.mean(bins == i) * abs(float(p[bins == i].mean())
                                 - float(y[bins == i].mean()))
        for i in range(10) if np.any(bins == i)
    ))


def _probability_metrics(y, p, prior):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    prior = np.asarray(prior, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(prior)
    y = y[valid].astype(int)
    p = np.clip(p[valid], 1e-6, 1.0 - 1e-6)
    prior = np.clip(prior[valid], 1e-6, 1.0 - 1e-6)
    result = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "mean_prediction": float(p.mean()) if len(y) else np.nan,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
        "ece": _ece(y, p),
        "baseline_brier": float(np.mean((prior - y) ** 2)) if len(y) else np.nan,
        "baseline_logloss": (
            float(log_loss(y, prior, labels=[0, 1])) if len(y) else np.nan),
        "baseline_ece": _ece(y, prior),
    }
    result["brier_delta"] = result["brier"] - result["baseline_brier"]
    result["logloss_delta"] = result["logloss"] - result["baseline_logloss"]
    if len(y) and np.unique(y).size == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    else:
        result["auc"] = result["average_precision"] = np.nan
    return result


def _benefit_metrics(y, p, prior):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    prior = np.asarray(prior, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(prior)
    y, p, prior = y[valid], p[valid], prior[valid]
    error = p - y
    prior_error = prior - y
    corr = (pd.Series(p).corr(pd.Series(y), method="spearman")
            if len(y) >= 3 else np.nan)
    result = {
        "n": int(len(y)),
        "mean_actual_bps": float(y.mean()) if len(y) else np.nan,
        "mean_predicted_bps": float(p.mean()) if len(y) else np.nan,
        "mae": float(np.mean(np.abs(error))) if len(y) else np.nan,
        "rmse": float(np.sqrt(np.mean(error ** 2))) if len(y) else np.nan,
        "bias": float(error.mean()) if len(y) else np.nan,
        "spearman": float(corr) if pd.notna(corr) else np.nan,
        "baseline_mae": (
            float(np.mean(np.abs(prior_error))) if len(y) else np.nan),
        "baseline_rmse": (
            float(np.sqrt(np.mean(prior_error ** 2))) if len(y) else np.nan),
    }
    result["mae_delta"] = result["mae"] - result["baseline_mae"]
    result["rmse_delta"] = result["rmse"] - result["baseline_rmse"]
    return result


def _reliability_rows(part, scenario, clock, h, slice_name, group):
    y = part.target.to_numpy(dtype=float)
    p = part.probability.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.minimum(np.searchsorted(edges, p, side="right") - 1, 9)
    return [{
        "scenario": scenario, "clock": clock, "h": h,
        "slice": slice_name, "group": group,
        "bin_left": edges[i], "bin_right": edges[i + 1],
        "n": int((bins == i).sum()),
        "predicted": float(p[bins == i].mean()),
        "actual": float(y[bins == i].mean()),
    } for i in range(10) if np.any(bins == i)]


def _benefit_bin_rows(part, scenario, clock, h, slice_name, group):
    valid = (
        np.isfinite(part.actual_future_bps)
        & np.isfinite(part.predicted_future_bps))
    work = part.loc[
        valid, ["actual_future_bps", "predicted_future_bps"]].copy()
    if work.empty:
        return []
    bins = min(10, len(work))
    work["bin"] = pd.qcut(
        work.predicted_future_bps.rank(method="first"),
        q=bins, labels=False)
    return [{
        "scenario": scenario, "clock": clock, "h": h,
        "slice": slice_name, "group": group, "bin": int(bin_id),
        "n": int(len(bin_part)),
        "predicted_mean_bps": float(bin_part.predicted_future_bps.mean()),
        "actual_mean_bps": float(bin_part.actual_future_bps.mean()),
        "calibration_gap_bps": float(
            bin_part.predicted_future_bps.mean()
            - bin_part.actual_future_bps.mean()),
    } for bin_id, bin_part in work.groupby("bin", sort=True)]


def _moving_block_interval(daily_delta, block, seed):
    values = np.asarray(daily_delta, dtype=float)
    n = len(values)
    if not n:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n_blocks))
    offsets = np.arange(block)
    ids = ((starts[:, :, None] + offsets[None, None, :]) % n)
    ids = ids.reshape(BOOTSTRAP_DRAWS, -1)[:, :n]
    draws = values[ids].mean(axis=1)
    return (float(np.quantile(draws, .025)),
            float(np.quantile(draws, .975)),
            float(np.mean(draws < 0.0)))


def _load_snapshots():
    frame = pd.read_csv(BASE / "snapshots.csv.gz")
    frame["valid_from"] = pd.to_datetime(frame.valid_from, utc=True)
    frame["source_at"] = pd.to_datetime(frame.source_at, utc=True)
    frame["snapshot_id"] = np.arange(len(frame), dtype=int)
    frame["dependent"] = (
        frame.source_kind.astype(str).isin(RECEIPT_DEPENDENT_SOURCE_KINDS)
        | frame.phase.astype(str).str.startswith("after_new_cbr")
    )
    return frame


def _query_grid(snapshots):
    local = snapshots.valid_from.dt.tz_convert(MOSCOW)
    first_available = max(
        snapshots.loc[snapshots.currency.eq(currency), "valid_from"].min()
        for currency in CORRIDORS)
    first_local = first_available.tz_convert(MOSCOW)
    first = first_local.date()
    if first_local.time() > dt.time(0, 15):
        first += dt.timedelta(days=1)
    last = local.dt.date.max()
    rows = []
    query_id = 0
    for day in pd.date_range(first, last, freq="D"):
        for clock in CLOCKS:
            hour, minute = map(int, clock.split(":"))
            stamp = dt.datetime.combine(
                day.date(), dt.time(hour, minute), tzinfo=MOSCOW)
            day_cutoff = dt.datetime.combine(
                day.date(), dt.time(0), tzinfo=MOSCOW) - dt.timedelta(
                    microseconds=1)
            for currency in CORRIDORS:
                rows.append({
                    "query_id": query_id,
                    "query_date": day.date(),
                    "year": day.year,
                    "weekday": day.day_name(),
                    "clock": clock,
                    "currency": currency,
                    "query_at": pd.Timestamp(stamp).tz_convert("UTC"),
                    "day_cutoff": pd.Timestamp(day_cutoff).tz_convert("UTC"),
                })
                query_id += 1
    return pd.DataFrame(rows)


def _asof_ids(queries, snapshots, left_on):
    result = pd.Series(np.nan, index=queries.index, dtype=float)
    for currency in CORRIDORS:
        left = queries.loc[
            queries.currency.eq(currency), ["query_id", left_on]
        ].sort_values(left_on)
        right = snapshots.loc[
            snapshots.currency.eq(currency), ["snapshot_id", "valid_from"]
        ].sort_values("valid_from")
        joined = pd.merge_asof(
            left, right, left_on=left_on, right_on="valid_from",
            direction="backward", allow_exact_matches=True)
        result.loc[joined.query_id.to_numpy(dtype=int)] = joined[
            "snapshot_id"].to_numpy(dtype=float)
    return result


def _selected_queries(queries, snapshots, scenario):
    if scenario == "calendar_assumed_replay":
        ids = _asof_ids(queries, snapshots, "query_at")
    elif scenario == "no_same_day_receipt":
        nondependent = snapshots.loc[~snapshots.dependent]
        dependent = snapshots.loc[snapshots.dependent]
        non_ids = _asof_ids(queries, nondependent, "query_at")
        past_ids = _asof_ids(queries, dependent, "day_cutoff")
        valid_times = snapshots.set_index("snapshot_id").valid_from
        non_times = pd.to_datetime(non_ids.map(valid_times), utc=True)
        past_times = pd.to_datetime(past_ids.map(valid_times), utc=True)
        use_past = non_ids.isna() | (past_times > non_times)
        ids = non_ids.where(~use_past, past_ids)
    else:
        raise ValueError(scenario)
    if ids.isna().any():
        raise AssertionError(f"{scenario}: query without a snapshot")
    chosen = snapshots.set_index("snapshot_id").loc[
        ids.astype(int).to_numpy()].reset_index()
    base = queries.reset_index(drop=True).copy()
    chosen = chosen.reset_index(drop=True)
    selected = pd.concat([
        base,
        chosen.drop(columns=["currency"]).add_prefix("snapshot_"),
    ], axis=1)
    selected["scenario"] = scenario
    if scenario == "no_same_day_receipt":
        same_day = (
            selected.snapshot_valid_from.dt.tz_convert(MOSCOW).dt.date
            == selected.query_date)
        if (same_day & selected.snapshot_dependent).any():
            raise AssertionError("same-day receipt-dependent row escaped gate")
    if not (selected.snapshot_valid_from <= selected.query_at).all():
        raise AssertionError("future snapshot selected")
    return selected


def _field(selected, field, h, default):
    column = f"snapshot_{field}_h{h}"
    if column not in selected:
        return selected[default]
    value = selected[column]
    present = value.notna() & value.astype(str).ne("")
    return value.where(present, selected[default])


def _target_arrays(series):
    output = {}
    for currency, item in series.items():
        for h in HORIZONS:
            fav = np.full(len(item), np.nan)
            benefit = np.full(len(item), np.nan)
            maturity = np.full(len(item), np.nan)
            for i in range(len(item)):
                y = target_now_favourable(item.values, i, h)
                b = benefit_forward_only(item.values, i, h)
                if y is not None:
                    fav[i] = y
                    benefit[i] = b
                    maturity[i] = item.dates[i + h].toordinal()
            output[(currency, h)] = (fav, benefit, maturity)
    return output


def _current_indices(selected, series):
    result = np.full(len(selected), -1, dtype=int)
    query_ord = np.asarray(
        [day.toordinal() for day in selected.query_date], dtype=int)
    for currency in CORRIDORS:
        mask = selected.currency.eq(currency).to_numpy()
        date_ord = np.asarray(
            [day.toordinal() for day in series[currency].dates], dtype=int)
        result[mask] = np.searchsorted(
            date_ord, query_ord[mask], side="right") - 1
    if (result < 0).any():
        raise AssertionError("query predates first effective CBR rate")
    return result


def _global_baseline_events(targets, h, kind):
    maturities, values = [], []
    for currency in CORRIDORS:
        fav, benefit, maturity = targets[(currency, h)]
        value = fav if kind == "probability" else benefit
        valid = np.isfinite(value) & np.isfinite(maturity)
        maturities.extend(maturity[valid].astype(int).tolist())
        values.extend(value[valid].astype(float).tolist())
    order = np.argsort(maturities)
    return (np.asarray(maturities, dtype=int)[order],
            np.asarray(values, dtype=float)[order])


def _past_baseline(query_dates, maturity, values):
    cutoff = np.asarray([
        day.toordinal() - EMBARGO_DAYS for day in query_dates
    ], dtype=int)
    count = np.searchsorted(maturity, cutoff, side="left")
    cumulative = np.cumsum(values)
    result = np.full(len(cutoff), np.nan)
    valid = count > 0
    result[valid] = cumulative[count[valid] - 1] / count[valid]
    return result, count


def _freshness(age, source_kind):
    age = np.asarray(age, dtype=float)
    kind = np.asarray(source_kind, dtype=str)
    market = np.isin(kind, [
        "moex_perpetual_prefix", "moex_early_prefix", "moex_prefix",
        "post_window_market", "post_receipt_perpetual", "post_receipt_market",
    ])
    history = kind == "cbr_history"
    fresh_limit = np.where(market, 90.0, np.where(history, 2160.0, 720.0))
    aging_limit = np.where(market, 240.0, np.where(history, 4320.0, 2160.0))
    return np.where(age <= fresh_limit, "fresh",
                    np.where(age <= aging_limit, "aging", "stale"))


def _scored_horizon(selected, series, targets, h):
    current = _current_indices(selected, series)
    target = np.full(len(selected), np.nan)
    actual_bps = np.full(len(selected), np.nan)
    for currency in CORRIDORS:
        mask = selected.currency.eq(currency).to_numpy()
        fav, benefit, _maturity = targets[(currency, h)]
        target[mask] = fav[current[mask]]
        actual_bps[mask] = benefit[current[mask]]
    probability = pd.to_numeric(
        selected[f"snapshot_probability_h{h}"], errors="coerce").to_numpy()
    predicted_bps = pd.to_numeric(
        selected[f"snapshot_expected_future_bps_h{h}"],
        errors="coerce").to_numpy()
    prob_maturity, prob_values = _global_baseline_events(
        targets, h, "probability")
    benefit_maturity, benefit_values = _global_baseline_events(
        targets, h, "benefit")
    prior, prior_n = _past_baseline(
        selected.query_date, prob_maturity, prob_values)
    prior_bps, prior_bps_n = _past_baseline(
        selected.query_date, benefit_maturity, benefit_values)
    source_at = pd.to_datetime(
        _field(selected, "source_at", h, "snapshot_source_at"), utc=True)
    source_kind = _field(
        selected, "source_kind", h, "snapshot_source_kind").astype(str)
    phase = _field(selected, "phase", h, "snapshot_phase").astype(str)
    confidence = _field(
        selected, "confidence", h, "snapshot_confidence").astype(str)
    benefit_source_at = pd.to_datetime(_field(
        selected, "benefit_source_at", h, "snapshot_source_at"), utc=True)
    benefit_source_kind = _field(
        selected, "benefit_source_kind", h,
        "snapshot_source_kind").astype(str)
    if (source_at > selected.query_at).any():
        raise AssertionError("future probability source selected")
    if (benefit_source_at > selected.query_at).any():
        raise AssertionError("future benefit source selected")
    age = (selected.query_at - source_at).dt.total_seconds().to_numpy() / 60.0
    benefit_age = (
        (selected.query_at - benefit_source_at).dt.total_seconds().to_numpy()
        / 60.0)
    return pd.DataFrame({
        "scenario": selected.scenario.to_numpy(),
        "query_id": selected.query_id.to_numpy(),
        "query_date": selected.query_date.to_numpy(),
        "year": selected.year.to_numpy(),
        "weekday": selected.weekday.to_numpy(),
        "clock": selected.clock.to_numpy(),
        "currency": selected.currency.to_numpy(),
        "query_at": selected.query_at.to_numpy(),
        "valid_from": selected.snapshot_valid_from.to_numpy(),
        "h": h,
        "target": target,
        "probability": probability,
        "prior_probability": prior,
        "prior_n": prior_n,
        "actual_future_bps": actual_bps,
        "predicted_future_bps": predicted_bps,
        "prior_future_bps": prior_bps,
        "prior_bps_n": prior_bps_n,
        "source_at": source_at.to_numpy(),
        "source_kind": source_kind.to_numpy(),
        "phase": phase.to_numpy(),
        "confidence": confidence.to_numpy(),
        "age_minutes": age,
        "freshness": _freshness(age, source_kind),
        "benefit_source_at": benefit_source_at.to_numpy(),
        "benefit_source_kind": benefit_source_kind.to_numpy(),
        "benefit_age_minutes": benefit_age,
        "benefit_freshness": _freshness(
            benefit_age, benefit_source_kind),
        "h1_known_after_receipt": (
            (h == 1) & phase.str.startswith("after_new_cbr").to_numpy()),
    })


def _metric_slices(scored):
    yield "ALL", "ALL", scored
    for currency, part in scored.groupby("currency", sort=True):
        yield "currency", currency, part
    for year, part in scored.groupby("year", sort=True):
        yield "year", str(year), part
    for (currency, year), part in scored.groupby(
            ["currency", "year"], sort=True):
        yield "currency_year", f"{currency}:{year}", part


def _bootstrap_rows(part, scenario, clock, h, scenario_i, _clock_i):
    rows = []
    probability_valid = (
        np.isfinite(part.target) & np.isfinite(part.probability)
        & np.isfinite(part.prior_probability))
    benefit_valid = (
        np.isfinite(part.actual_future_bps)
        & np.isfinite(part.predicted_future_bps)
        & np.isfinite(part.prior_future_bps))
    deltas = {}
    if probability_valid.any():
        work = part.loc[probability_valid].copy()
        work["delta"] = (
            (work.probability - work.target) ** 2
            - (work.prior_probability - work.target) ** 2)
        deltas["brier"] = work.groupby("query_date").delta.mean()
    if benefit_valid.any():
        work = part.loc[benefit_valid].copy()
        work["delta"] = (
            (work.predicted_future_bps - work.actual_future_bps).abs()
            - (work.prior_future_bps - work.actual_future_bps).abs())
        deltas["benefit_mae"] = work.groupby("query_date").delta.mean()
    for metric_i, (metric, daily) in enumerate(deltas.items()):
        for block in BLOCKS:
            # The same held daily-loss path must receive the same Monte Carlo
            # interval at every display clock. Clock-specific seeds could make
            # an identical state straddle zero at one clock but not another.
            seed = (20260906 + scenario_i * 1_000_000
                    + h * 100 + metric_i * 10 + block)
            low, high, probability_better = _moving_block_interval(
                daily.to_numpy(), block, seed)
            rows.append({
                "scenario": scenario, "clock": clock, "h": h,
                "metric": metric, "block_dates": block,
                "n_dates": int(len(daily)),
                "mean_delta": float(daily.mean()),
                "ci_low": low, "ci_high": high,
                "probability_router_better": probability_better,
            })
    return rows


def run_audit():
    snapshots = _load_snapshots()
    queries = _query_grid(snapshots)
    series = load("data/cbr_rates_2010_2026.json")
    targets = _target_arrays(series)
    selected = {
        scenario: _selected_queries(queries, snapshots, scenario)
        for scenario in SCENARIOS
    }

    probability_metrics, benefit_metrics = [], []
    reliability, benefit_bins, bootstrap, coverage = [], [], [], []
    sample = []
    for scenario_i, scenario in enumerate(SCENARIOS):
        for h in HORIZONS:
            scored = _scored_horizon(selected[scenario], series, targets, h)
            if h == 5:
                sample.append(scored[
                    scored.query_date.eq(scored.query_date.max())
                    & scored.clock.isin(["09:15", "17:45", "18:45", "21:15"])
                ])
            for clock_i, (clock, part) in enumerate(
                    scored.groupby("clock", sort=False)):
                for slice_name, group, sliced in _metric_slices(part):
                    prefix = {
                        "scenario": scenario, "clock": clock, "h": h,
                        "slice": slice_name, "group": group,
                    }
                    probability_metrics.append({
                        **prefix,
                        **_probability_metrics(
                            sliced.target, sliced.probability,
                            sliced.prior_probability),
                    })
                    benefit_metrics.append({
                        **prefix,
                        **_benefit_metrics(
                            sliced.actual_future_bps,
                            sliced.predicted_future_bps,
                            sliced.prior_future_bps),
                    })
                reliability.extend(_reliability_rows(
                    part, scenario, clock, h, "ALL", "ALL"))
                benefit_bins.extend(_benefit_bin_rows(
                    part, scenario, clock, h, "ALL", "ALL"))
                for currency, currency_part in part.groupby(
                        "currency", sort=True):
                    reliability.extend(_reliability_rows(
                        currency_part, scenario, clock, h,
                        "currency", currency))
                    benefit_bins.extend(_benefit_bin_rows(
                        currency_part, scenario, clock, h,
                        "currency", currency))
                bootstrap.extend(_bootstrap_rows(
                    part, scenario, clock, h, scenario_i, clock_i))
                coverage.extend(
                    part.groupby([
                        "year", "weekday", "currency",
                        "source_kind", "phase", "confidence", "freshness",
                        "benefit_source_kind", "benefit_freshness",
                    ], dropna=False).size().rename("rows").reset_index().assign(
                        scenario=scenario, clock=clock, h=h).to_dict("records"))

    probability_metrics = pd.DataFrame(probability_metrics)
    benefit_metrics = pd.DataFrame(benefit_metrics)
    reliability = pd.DataFrame(reliability)
    benefit_bins = pd.DataFrame(benefit_bins)
    bootstrap = pd.DataFrame(bootstrap)
    coverage = pd.DataFrame(coverage)
    sample = pd.concat(sample, ignore_index=True)

    overall_p = probability_metrics[
        probability_metrics.slice.eq("ALL")].copy()
    overall_b = benefit_metrics[benefit_metrics.slice.eq("ALL")].copy()
    brier_ci = bootstrap[bootstrap.metric.eq("brier")].groupby(
        ["scenario", "clock", "h"]).ci_high.max().rename("brier_ci_high")
    benefit_ci = bootstrap[bootstrap.metric.eq("benefit_mae")].groupby(
        ["scenario", "clock", "h"]).ci_high.max().rename(
            "benefit_mae_ci_high")
    weak = overall_p.merge(
        overall_b, on=["scenario", "clock", "h", "slice", "group"],
        suffixes=("_probability", "_benefit"),
    ).merge(brier_ci, on=["scenario", "clock", "h"]).merge(
        benefit_ci, on=["scenario", "clock", "h"])
    weak["probability_supported"] = weak.brier_ci_high < 0.0
    weak["benefit_supported"] = weak.benefit_mae_ci_high < 0.0
    weak["flag_high_ece"] = weak.ece > .08
    weak["flag_low_auc"] = weak.auc < .55
    weak["flag_small_n"] = (
        (weak.n_probability < 100) | (weak.n_benefit < 100))
    weak["flag_probability_not_supported"] = ~weak.probability_supported
    weak["flag_benefit_not_supported"] = ~weak.benefit_supported
    slice_weak = probability_metrics.merge(
        benefit_metrics,
        on=["scenario", "clock", "h", "slice", "group"],
        suffixes=("_probability", "_benefit"),
    )
    slice_weak["flag_high_ece"] = slice_weak.ece > .08
    slice_weak["flag_low_auc"] = slice_weak.auc < .55
    slice_weak["flag_small_n"] = (
        (slice_weak.n_probability < 100)
        | (slice_weak.n_benefit < 100))
    slice_weak["flag_brier_worse_point"] = slice_weak.brier_delta >= 0.0
    slice_weak["flag_benefit_worse_point"] = slice_weak.mae_delta >= 0.0

    after = pd.read_csv(PANEL)
    after.date = pd.to_datetime(after.date).dt.date
    alignment_checked = 0
    for currency in CORRIDORS:
        item = series[currency]
        date_ord = np.asarray([day.toordinal() for day in item.dates])
        part = after[after.currency.eq(currency)]
        expected = np.searchsorted(
            date_ord,
            np.asarray([day.toordinal() for day in part.date]),
            side="right") - 1
        if not np.array_equal(expected, part.current_index.to_numpy(dtype=int)):
            raise AssertionError("announcement current-index alignment failed")
        alignment_checked += len(part)

    checks = {
        "query_rows_per_scenario": int(len(queries)),
        "first_query_date": str(queries.query_date.min()),
        "last_query_date": str(queries.query_date.max()),
        "clocks": list(CLOCKS),
        "horizons": list(HORIZONS),
        "all_queries_have_snapshot": True,
        "selected_valid_from_not_later_than_query": True,
        "horizon_source_at_not_later_than_query": True,
        "no_same_day_receipt_gate_verified": True,
        "announcement_current_index_rows_verified": alignment_checked,
        "baseline_unique_events_and_strict_maturity": True,
        "open_diagnostic_only": True,
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
    }
    return {
        "probability_metrics": probability_metrics,
        "benefit_metrics": benefit_metrics,
        "reliability": reliability,
        "benefit_bins": benefit_bins,
        "bootstrap": bootstrap,
        "coverage": coverage,
        "weak": weak,
        "slice_weak": slice_weak,
        "sample": sample,
        "checks": checks,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_audit()
    result["probability_metrics"].to_csv(
        OUT / "probability_metrics.csv", index=False)
    result["benefit_metrics"].to_csv(
        OUT / "benefit_metrics.csv", index=False)
    result["reliability"].to_csv(OUT / "reliability_bins.csv", index=False)
    result["benefit_bins"].to_csv(
        OUT / "benefit_calibration_bins.csv", index=False)
    result["bootstrap"].to_csv(OUT / "paired_bootstrap.csv", index=False)
    result["coverage"].to_csv(OUT / "coverage.csv", index=False)
    result["weak"].to_csv(OUT / "weak_spots.csv", index=False)
    result["slice_weak"].to_csv(
        OUT / "slice_weak_spots.csv", index=False)
    result["sample"].to_csv(OUT / "selection_trace_sample.csv", index=False)
    (OUT / "audit_checks.json").write_text(json.dumps(
        result["checks"], ensure_ascii=False, indent=2))

    sources = [
        BASE / "metadata.json", BASE / "snapshots.csv.gz", PANEL,
        Path("data/cbr_rates_2010_2026.json"), REGISTERED,
        Path("research/temperature_t19_anytime_quality_audit.py"),
    ]
    weak = result["weak"]
    metadata = {
        "packet": "temperature-T19",
        "selection_or_refit": False,
        "opened_period_diagnostic_only": True,
        "query_rows_per_scenario": result["checks"]["query_rows_per_scenario"],
        "state_horizon_rows": int(len(weak)),
        "probability_supported_states": int(weak.probability_supported.sum()),
        "benefit_supported_states": int(weak.benefit_supported.sum()),
        "currency_year_high_ece_rows": int((
            result["slice_weak"].slice.eq("currency_year")
            & result["slice_weak"].flag_high_ece).sum()),
        "historical_receipts_certified": False,
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({
        **metadata,
        "source_sha256": "saved in metadata.json",
        "probability_support_by_scenario": weak.groupby(
            "scenario").probability_supported.agg(["sum", "count"]).to_dict(
                "index"),
        "benefit_support_by_scenario": weak.groupby(
            "scenario").benefit_supported.agg(["sum", "count"]).to_dict(
                "index"),
        "highest_ece": weak.sort_values("ece", ascending=False).head(5)[
            ["scenario", "clock", "h", "ece", "auc", "brier_delta"]
        ].to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
