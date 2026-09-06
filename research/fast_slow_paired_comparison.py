"""Paired fast (15:30) versus slow (after receipt) case comparison."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import HORIZONS, build_targets
from research.after_publication_ap1 import adjusted
from research.after_publication_ap37_effective import CANDIDATE
from research.round5_features import load_round5_features
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.round6_multihorizon_policy_screen import PolicySpec, fire
from research.round6_uzbek_central_bank_models import _forward


OUT = Path("results/research/fast_slow_paired")
FAST = Path("results/research/round6/fixing_availability_router/outputs.pkl")
SLOW = Path("results/research/after_publication/ap37_effective")
REGISTERED = Path("research/fast_slow_paired_registered.md")
FAST_NAME = "15:30 availability_route"
SLOW_NAME = "after-receipt AP37"
YEARS = (2025, 2026)
SPEC = PolicySpec("rolling", .22, 20)
SEED = 20260906
N_BOOTSTRAP = 1000


def _fast_frame():
    _x, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    with FAST.open("rb") as handle:
        output = pickle.load(handle)["availability_route"]
    _valid, signal = fire(
        output, YEARS, dates, currencies, targets["fav_h5"], SPEC)
    test = np.zeros(len(index), dtype=bool)
    for year in YEARS:
        test[np.asarray(output[year]["test_idx"], dtype=int)] = True
    data = {
        "currency": currencies[test],
        "date": dates[test],
        "fast_signal": signal[test],
    }
    for h in HORIZONS:
        data[f"now_h{h}"] = targets[f"fav_h{h}"][test]
        data[f"closing_h{h}"] = targets[f"close_h{h}"][test]
        data[f"symmetric_h{h}"] = targets[f"benefit_h{h}"][test]
        data[f"forward_h{h}"] = forwards[h][test]
    return pd.DataFrame(data)


def _slow_frame():
    panel = pd.read_csv(SLOW / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    with np.load(SLOW / "outputs.npz") as source:
        slow = source["signal__" + CANDIDATE].astype(bool)
        later = source["later"].astype(bool)
        data = {
            "currency": panel.currency[later].to_numpy(),
            "date": panel.date[later].to_numpy(),
            "slow_signal": slow[later],
        }
        for h in HORIZONS:
            data[f"slow_now_h{h}"] = source[f"y{h}"][later]
            data[f"slow_symmetric_h{h}"] = source[f"sym{h}"][later]
            data[f"slow_forward_h{h}"] = source[f"forward{h}"][later]
    return pd.DataFrame(data)


def common_support():
    frame = _fast_frame().merge(
        _slow_frame(), on=["currency", "date"], how="inner", validate="one_to_one"
    ).sort_values(["date", "currency"]).reset_index(drop=True)
    if set(frame.currency) != set(CORRIDORS):
        raise AssertionError("common support lost a corridor")
    for h in HORIZONS:
        both = np.isfinite(frame[f"now_h{h}"]) & np.isfinite(
            frame[f"slow_now_h{h}"])
        if not np.array_equal(
                frame.loc[both, f"now_h{h}"].to_numpy(),
                frame.loc[both, f"slow_now_h{h}"].to_numpy()):
            raise AssertionError(f"target mismatch at h={h}")
        for name in ("symmetric", "forward"):
            left = frame[f"{name}_h{h}"].to_numpy()
            right = frame[f"slow_{name}_h{h}"].to_numpy()
            valid = np.isfinite(left) & np.isfinite(right)
            if not np.allclose(left[valid], right[valid], atol=0, rtol=0):
                raise AssertionError(f"{name} mismatch at h={h}")
    return frame


def _weighted_mean(values, selected, weights):
    valid = selected & np.isfinite(values)
    denominator = float(np.sum(weights[valid]))
    return (float(np.sum(weights[valid] * values[valid]) / denominator)
            if denominator > 0 else np.nan)


def _metric(frame, signal, target, valid, symmetric, forward):
    active = valid & signal
    dates = frame.date.to_numpy()
    currencies = frame.currency.to_numpy()
    groups = pd.factorize(np.asarray([
        f"{currency}-{day.year}" for currency, day in zip(currencies, dates)
    ], dtype=object))[0]
    hit = float(np.mean(target[active])) if active.any() else np.nan
    base = float(np.mean(target[valid])) if valid.any() else np.nan
    corridor_lifts = []
    for currency in CORRIDORS:
        scope = valid & (currencies == currency)
        chosen = active & (currencies == currency)
        if scope.any() and chosen.any() and np.mean(target[scope]) > 0:
            corridor_lifts.append(
                float(np.mean(target[chosen]) / np.mean(target[scope])))
    return {
        "n_scope": int(valid.sum()),
        "n_signals": int(active.sum()),
        "hit_rate": hit,
        "random_day_rate": base,
        "pooled_lift": hit / base if base > 0 else np.nan,
        "adjusted_lift": adjusted(target, signal, valid, groups),
        "corridor_lift_min": min(corridor_lifts),
        "corridor_lift_max": max(corridor_lifts),
        "signals_per_currency_week": rate_per_week(
            int(active.sum()), len(CORRIDORS), dates, valid),
        "symmetric_bps": float(np.nanmean(symmetric[active])),
        "future_only_bps": float(np.nanmean(forward[active])),
    }


def scorecard(frame):
    rows = []
    for h in HORIZONS:
        symmetric = frame[f"symmetric_h{h}"].to_numpy(float)
        forward = frame[f"forward_h{h}"].to_numpy(float)
        for target_name, target in (
            ("now_favourable", frame[f"now_h{h}"].to_numpy(float)),
            ("window_closing", frame[f"closing_h{h}"].to_numpy(float)),
        ):
            valid = (
                np.isfinite(target)
                & np.isfinite(frame[f"slow_now_h{h}"].to_numpy(float))
                & np.isfinite(symmetric)
                & np.isfinite(forward)
            )
            for candidate, column in (
                (FAST_NAME, "fast_signal"), (SLOW_NAME, "slow_signal"),
            ):
                rows.append({
                    "candidate": candidate,
                    "target": target_name,
                    "h": h,
                    **_metric(frame, frame[column].to_numpy(bool), target,
                              valid, symmetric, forward),
                })
    return pd.DataFrame(rows)


def paired_bootstrap(frame):
    dates = frame.date.to_numpy()
    currencies = frame.currency.to_numpy()
    groups = pd.factorize(np.asarray([
        f"{currency}-{day.year}" for currency, day in zip(currencies, dates)
    ], dtype=object))[0]
    unique_dates, date_id = np.unique(dates, return_inverse=True)
    n_dates = len(unique_dates)
    block = 20
    rng = np.random.default_rng(SEED)
    rows = []
    for h in HORIZONS:
        symmetric = frame[f"symmetric_h{h}"].to_numpy(float)
        forward = frame[f"forward_h{h}"].to_numpy(float)
        for target_name, target in (
            ("now_favourable", frame[f"now_h{h}"].to_numpy(float)),
            ("window_closing", frame[f"closing_h{h}"].to_numpy(float)),
        ):
            valid = (
                np.isfinite(target)
                & np.isfinite(frame[f"slow_now_h{h}"].to_numpy(float))
                & np.isfinite(symmetric)
                & np.isfinite(forward)
            )
            draws = {name: np.empty(N_BOOTSTRAP) for name in (
                "lift", "symmetric", "future")}
            for draw in range(N_BOOTSTRAP):
                starts = rng.integers(0, n_dates, size=int(np.ceil(n_dates / block)))
                picked = ((starts[:, None] + np.arange(block)) % n_dates).ravel()
                picked = picked[:n_dates]
                day_weights = np.bincount(picked, minlength=n_dates)
                weights = day_weights[date_id].astype(float)
                fast = frame.fast_signal.to_numpy(bool)
                slow = frame.slow_signal.to_numpy(bool)
                draws["lift"][draw] = (
                    adjusted(target, slow, valid, groups, weights)
                    - adjusted(target, fast, valid, groups, weights)
                )
                draws["symmetric"][draw] = (
                    _weighted_mean(symmetric, valid & slow, weights)
                    - _weighted_mean(symmetric, valid & fast, weights)
                )
                draws["future"][draw] = (
                    _weighted_mean(forward, valid & slow, weights)
                    - _weighted_mean(forward, valid & fast, weights)
                )
            for metric, values in draws.items():
                rows.append({
                    "target": target_name, "h": h, "metric": metric,
                    "slow_minus_fast_mean": float(np.nanmean(values)),
                    "ci_lo": float(np.nanquantile(values, .025)),
                    "ci_hi": float(np.nanquantile(values, .975)),
                    "draws": N_BOOTSTRAP,
                    "block": "20 calendar decision dates; all currencies together",
                })
    return pd.DataFrame(rows)


def _last_close(rows, day, cutoff):
    chosen = [row for row in rows
              if row["begin"].date() == day
              and row["begin"].time() >= pd.Timestamp("10:00").time()
              and row["end"].time() < cutoff]
    return float(chosen[-1]["close"]) if chosen else np.nan


def waiting_cost(frame):
    history, _digest = load_spot_1530_history()
    rows = history["CNYRUB_TOM"]
    by_date = {}
    for day in sorted(set(frame.date)):
        early = _last_close(rows, day, pd.Timestamp("15:30").time())
        late = _last_close(rows, day, pd.Timestamp("18:10").time())
        if np.isfinite(early) and np.isfinite(late):
            by_date[day] = (late / early - 1.0) * 10000.0
    daily = pd.DataFrame({"date": list(by_date), "cny_wait_bps": list(by_date.values())})
    activity = frame.groupby("date").agg(
        fast_any=("fast_signal", "max"), slow_any=("slow_signal", "max"),
    ).reset_index()
    daily = daily.merge(activity, on="date", how="left")
    summary = []
    for scope, mask in (
        ("all_common_dates", np.ones(len(daily), dtype=bool)),
        ("at_least_one_fast_signal", daily.fast_any.to_numpy(bool)),
        ("at_least_one_slow_signal", daily.slow_any.to_numpy(bool)),
    ):
        values = daily.loc[mask, "cny_wait_bps"].to_numpy(float)
        summary.append({
            "scope": scope,
            "n_dates": len(values),
            "mean_signed_bps": float(np.mean(values)),
            "median_signed_bps": float(np.median(values)),
            "mean_absolute_bps": float(np.mean(np.abs(values))),
            "p90_absolute_bps": float(np.quantile(np.abs(values), .90)),
            "positive_move_share": float(np.mean(values > 0)),
            "meaning": "positive = CNY cost more RUB by 18:10 than at 15:30",
        })
    return daily, pd.DataFrame(summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = common_support()
    frame.to_csv(OUT / "common_support.csv", index=False)
    scores = scorecard(frame)
    scores.to_csv(OUT / "scorecard.csv", index=False)
    bootstrap = paired_bootstrap(frame)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    overlap = pd.crosstab(frame.fast_signal, frame.slow_signal).rename_axis(
        "fast_signal").rename_axis("slow_signal", axis=1)
    overlap.to_csv(OUT / "signal_overlap.csv")
    daily, waiting = waiting_cost(frame)
    daily.to_csv(OUT / "cny_waiting_cost_by_date.csv", index=False)
    waiting.to_csv(OUT / "cny_waiting_cost_summary.csv", index=False)
    sources = [FAST, SLOW / "outputs.npz", SLOW / "announcement_panel.csv",
               REGISTERED]
    metadata = {
        "packet": "fast-slow-paired",
        "common_rows": int(len(frame)),
        "common_dates": int(frame.date.nunique()),
        "years": list(YEARS),
        "fast": FAST_NAME,
        "slow": SLOW_NAME,
        "targets_exact_on_common_support": True,
        "official_cbr_intraday_waiting_cost": (
            "not identifiable: same today-effective reference at both clocks"
        ),
        "market_waiting_proxy": "CNYRUB_TOM 15:30 to 18:10 completed closes",
        "bank_execution_validated": False,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print("\nNOW FAVOURABLE\n" + scores[
        scores.target.eq("now_favourable")
    ].to_string(index=False))
    print("\nOVERLAP\n" + overlap.to_string())
    print("\nWAITING\n" + waiting.to_string(index=False))


if __name__ == "__main__":
    main()
