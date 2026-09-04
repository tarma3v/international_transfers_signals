#!/usr/bin/env python3
"""Causal audit of the ``version_b`` logistic-regression benchmark.

The teammate benchmark selects the top score share inside each completed test
fold.  That is useful for measuring ranking quality, but the resulting alert
rule cannot run online because the future test-score distribution is unknown.

This script imports the exact feature/model implementation from a checkout of
``version_b`` and compares that diagnostic with causal alternatives.  It also
runs a stricter 30-month fit / 6-month calibration / 6-month test protocol in
which the same fitted model scores both calibration and test rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON = 5
TARGET = "target_fav"
MODEL = "logistic_regression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-b", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def import_version_b(path: Path):
    sys.path.insert(0, str((path / "src").resolve()))
    from transfer_lift.data import CORRIDORS, load_rates
    from transfer_lift.evaluation import make_walk_forward_folds, purge_label_overlap
    from transfer_lift.features import build_dataset, feature_columns
    from transfer_lift.models import make_model

    return (
        CORRIDORS,
        load_rates,
        make_walk_forward_folds,
        purge_label_overlap,
        build_dataset,
        feature_columns,
        make_model,
    )


def fit_score_source(frame, features, folds, purge_label_overlap, make_model):
    """Exact 36-month fits used by version_b, retaining row-level scores."""
    chunks = []
    train_cutoffs: dict[tuple[int, str, float], float] = {}
    for fold_id, fold in enumerate(folds):
        train = frame[frame.date.between(fold.train_start, fold.train_end)].copy()
        train = purge_label_overlap(train, HORIZON)
        test = frame[frame.date.between(fold.test_start, fold.test_end)].copy()
        model = make_model(MODEL).fit(train, TARGET, features)
        train["score"] = model.score(train, features)
        test["score"] = model.score(test, features)
        test["fold"] = fold_id
        for corridor, part in train.groupby("corridor"):
            for rate in (.15, .20, .30, .40):
                train_cutoffs[(fold_id, corridor, rate)] = float(
                    part.score.quantile(1.0 - rate)
                )
        chunks.append(test)
    return pd.concat(chunks, ignore_index=True), train_cutoffs


def fit_score_nested(frame, features, folds, purge_label_overlap, make_model):
    """Strict 30m fit / 6m untouched calibration / 6m test scores."""
    test_chunks, calibration_scores = [], {}
    for fold_id, fold in enumerate(folds):
        calibration_start = fold.test_start - pd.DateOffset(months=6)
        fit_end = calibration_start - pd.Timedelta(days=1)
        fit = frame[frame.date.between(fold.train_start, fit_end)].copy()
        fit = purge_label_overlap(fit, HORIZON)
        calibration = frame[
            frame.date.between(calibration_start, fold.train_end)
        ].copy()
        test = frame[frame.date.between(fold.test_start, fold.test_end)].copy()
        model = make_model(MODEL).fit(fit, TARGET, features)
        calibration["score"] = model.score(calibration, features)
        test["score"] = model.score(test, features)
        test["fold"] = fold_id
        for corridor, part in calibration.groupby("corridor"):
            calibration_scores[(fold_id, corridor)] = part.sort_values("date")[
                "score"
            ].to_numpy()
        test_chunks.append(test)
    return pd.concat(test_chunks, ignore_index=True), calibration_scores


def future_top_mask(scored: pd.DataFrame, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Original non-operational top-K selection inside the complete test fold."""
    active = np.zeros(len(scored), dtype=bool)
    for _, part in scored.groupby(["corridor", "fold"], sort=False):
        n = max(1, int(np.ceil(len(part) * rate)))
        active[part.nlargest(n, "score").index] = True
    return active, np.ones(len(scored), dtype=bool)


def train_cutoff_mask(scored, cutoffs, rate):
    active = np.zeros(len(scored), dtype=bool)
    for (fold, corridor), part in scored.groupby(["fold", "corridor"]):
        cutoff = cutoffs[(int(fold), str(corridor), rate)]
        active[part.index] = part.score.to_numpy() >= cutoff
    return active, np.ones(len(scored), dtype=bool)


def past_oof_mask(scored, rate: float, mode: str, window: int | None = None):
    """Causal thresholds based only on scores from completed/past test rows.

    Fold zero seeds score history and is deliberately outside the evaluation
    scope.  No target values are used for thresholding.
    """
    active = np.zeros(len(scored), dtype=bool)
    scope = np.zeros(len(scored), dtype=bool)
    for _, currency_part in scored.groupby("corridor", sort=False):
        history: list[float] = []
        for fold, part in currency_part.groupby("fold", sort=True):
            part = part.sort_values("date")
            if int(fold) == 0:
                history.extend(part.score.astype(float).tolist())
                continue
            scope[part.index] = True
            if mode == "previous_fold":
                previous = currency_part[currency_part.fold == int(fold) - 1].score
                cutoff = float(previous.quantile(1.0 - rate))
                active[part.index] = part.score.to_numpy() >= cutoff
                history.extend(part.score.astype(float).tolist())
            elif mode == "expanding":
                cutoff = float(np.quantile(history, 1.0 - rate))
                active[part.index] = part.score.to_numpy() >= cutoff
                history.extend(part.score.astype(float).tolist())
            elif mode == "rolling":
                for idx, value in zip(part.index, part.score.astype(float)):
                    reference = history[-window:] if window else history
                    cutoff = float(np.quantile(reference, 1.0 - rate))
                    active[idx] = value >= cutoff
                    history.append(value)
            else:
                raise ValueError(mode)
    return active, scope


def nested_mask(scored, calibration_scores, rate: float, rolling: int | None):
    active = np.zeros(len(scored), dtype=bool)
    for (fold, corridor), part in scored.groupby(["fold", "corridor"]):
        history = list(calibration_scores[(int(fold), str(corridor))])
        for idx, value in zip(part.sort_values("date").index, part.sort_values("date").score):
            reference = history[-rolling:] if rolling else history
            cutoff = float(np.quantile(reference, 1.0 - rate))
            active[idx] = float(value) >= cutoff
            if rolling:
                history.append(float(value))
    return active, np.ones(len(scored), dtype=bool)


def rate_per_currency_week(scored, active, scope, corridors) -> float:
    rates = []
    for corridor in corridors:
        cm = scope & scored.corridor.eq(corridor).to_numpy()
        if not cm.any():
            continue
        span = max((scored.loc[cm, "date"].max() - scored.loc[cm, "date"].min()).days / 7, 1)
        rates.append(float((active & cm).sum() / span))
    return float(np.mean(rates))


def summarize(name, scored, active, scope, corridors):
    valid = scope & scored[TARGET].notna().to_numpy()
    fired = valid & active
    base = float(scored.loc[valid, TARGET].mean())
    hit = float(scored.loc[fired, TARGET].mean()) if fired.any() else np.nan
    corridor_lifts = []
    for corridor in corridors:
        cm = valid & scored.corridor.eq(corridor).to_numpy()
        ca = fired & scored.corridor.eq(corridor).to_numpy()
        if cm.any() and ca.any():
            corridor_lifts.append(
                float(scored.loc[ca, TARGET].mean() / scored.loc[cm, TARGET].mean())
            )
    return {
        "policy": name,
        "n_test": int(valid.sum()),
        "n_signals": int(fired.sum()),
        "signal_share": float(fired.sum() / valid.sum()),
        "frequency": rate_per_currency_week(scored, fired, valid, corridors),
        "base_rate": base,
        "hit_rate": hit,
        "aggregate_lift": hit / base,
        "mean_corridor_lift": float(np.mean(corridor_lifts)),
        "min_corridor_lift": float(np.min(corridor_lifts)),
        "benefit_bps": float(scored.loc[fired, "benefit_bps"].mean()),
    }


def annual_rows(policy, scored, active, scope, corridors):
    rows = []
    years = sorted(scored.loc[scope, "date"].dt.year.unique())
    for year in years:
        ym = scope & scored.date.dt.year.eq(year).to_numpy()
        if not ym.any():
            continue
        row = summarize(policy, scored, active & ym, ym, corridors)
        row["year"] = int(year)
        rows.append(row)
    return rows


def block_bootstrap(policy, scored, active, scope, draws: int = 4000):
    """Four-week moving-block interval, keeping currencies/dates together."""
    valid = scope & scored[TARGET].notna().to_numpy()
    fired = valid & active
    z = scored.loc[valid, ["date", TARGET, "benefit_bps"]].copy()
    z["fired"] = fired[valid]
    z["week"] = z.date.dt.to_period("W-SUN").astype(str)
    weekly = []
    for _, part in z.groupby("week", sort=True):
        signal = part.fired.to_numpy(bool)
        benefits = part.loc[signal, "benefit_bps"].dropna()
        weekly.append(
            [
                float(part[TARGET].sum()),
                float(len(part)),
                float(part.loc[signal, TARGET].sum()),
                float(signal.sum()),
                float(benefits.sum()),
                float(len(benefits)),
            ]
        )
    stats = np.asarray(weekly)
    block = 4
    n_blocks = int(np.ceil(len(stats) / block))
    rng = np.random.default_rng(20260904)
    lift_draws, benefit_draws = [], []
    for _ in range(draws):
        starts = rng.integers(0, len(stats) - block + 1, size=n_blocks)
        pick = np.concatenate([np.arange(start, start + block) for start in starts])[
            : len(stats)
        ]
        y_sum, n, hit_sum, n_signal, benefit_sum, n_benefit = stats[pick].sum(axis=0)
        if n_signal > 0 and y_sum > 0:
            lift_draws.append((hit_sum / n_signal) / (y_sum / n))
        if n_benefit > 0:
            benefit_draws.append(benefit_sum / n_benefit)
    lifts = np.asarray(lift_draws)
    benefits = np.asarray(benefit_draws)
    return {
        "policy": policy,
        "draws": draws,
        "lift_ci_low": float(np.quantile(lifts, .025)),
        "lift_ci_high": float(np.quantile(lifts, .975)),
        "p_lift_le_1": float((np.sum(lifts <= 1.0) + 1) / (len(lifts) + 1)),
        "p_lift_le_1_30": float((np.sum(lifts <= 1.30) + 1) / (len(lifts) + 1)),
        "benefit_ci_low": float(np.quantile(benefits, .025)),
        "benefit_ci_high": float(np.quantile(benefits, .975)),
    }


def main() -> None:
    args = parse_args()
    (
        corridors,
        load_rates,
        make_walk_forward_folds,
        purge_label_overlap,
        build_dataset,
        feature_columns,
        make_model,
    ) = import_version_b(args.version_b)
    frame = build_dataset(load_rates(args.data), horizon=HORIZON)
    features = feature_columns(frame, target_col=TARGET)
    folds = make_walk_forward_folds(frame, train_months=36, test_months=6, step_months=6)
    source, train_cutoffs = fit_score_source(
        frame, features, folds, purge_label_overlap, make_model
    )
    nested, calibration_scores = fit_score_nested(
        frame, features, folds, purge_label_overlap, make_model
    )

    policies: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] = []
    active, scope = future_top_mask(source, .15)
    policies.append(("future_test_top15_noncausal", source, active, scope))
    for rate in (.15, .20, .30, .40):
        active, scope = train_cutoff_mask(source, train_cutoffs, rate)
        policies.append((f"train_score_q{int(rate*100)}", source, active, scope))
    for rate in (.15, .20, .30, .40):
        for mode, window in (("previous_fold", None), ("expanding", None), ("rolling", 120), ("rolling", 250)):
            active, scope = past_oof_mask(source, rate, mode, window)
            suffix = f"_{window}" if window else ""
            policies.append((f"past_oof_{mode}{suffix}_q{int(rate*100)}", source, active, scope))
    for rate in (.15, .20, .30, .40):
        for rolling in (None, 120, 250):
            active, scope = nested_mask(nested, calibration_scores, rate, rolling)
            suffix = "fixed" if rolling is None else f"rolling_{rolling}"
            policies.append((f"nested_calib_{suffix}_q{int(rate*100)}", nested, active, scope))

    summary = []
    annual = []
    periods = []
    bootstrap = []
    for name, scored, active, scope in policies:
        summary.append(summarize(name, scored, active, scope, corridors))
        annual.extend(annual_rows(name, scored, active, scope, corridors))
        if name in {
            "future_test_top15_noncausal",
            "past_oof_expanding_q20",
            "past_oof_rolling_120_q30",
            "nested_calib_fixed_q20",
            "nested_calib_rolling_120_q20",
        }:
            bootstrap.append(block_bootstrap(name, scored, active, scope))
        for period, first_year, last_year in (
            ("2022-2023", 2022, 2023),
            ("2024-2026", 2024, 2026),
        ):
            period_scope = scope & scored.date.dt.year.between(first_year, last_year).to_numpy()
            if period_scope.any():
                row = summarize(name, scored, active & period_scope, period_scope, corridors)
                row["period"] = period
                periods.append(row)
    summary_frame = pd.DataFrame(summary)
    annual_frame = pd.DataFrame(annual)
    period_frame = pd.DataFrame(periods)
    bootstrap_frame = pd.DataFrame(bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output / "summary.csv", index=False)
    annual_frame.to_csv(args.output / "annual.csv", index=False)
    period_frame.to_csv(args.output / "periods.csv", index=False)
    bootstrap_frame.to_csv(args.output / "bootstrap.csv", index=False)
    protocol = {
        "source_commit": "aa44f10a47bb9bd72379331bd4596eab3c4944b0",
        "data": str(args.data.resolve()),
        "rows": int(len(frame)),
        "folds": len(folds),
        "first_test": str(source.date.min().date()),
        "last_test": str(source.date.max().date()),
        "target": TARGET,
        "horizon_publications": HORIZON,
        "model": MODEL,
        "source_fit": "36 months, h=5 purged, 6 month test",
        "nested_fit": "30 months fit, h=5 purged, 6 months untouched calibration, 6 months test",
        "warning": "All results are retrospective because the full period was already inspected.",
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    show = summary_frame[
        summary_frame.policy.isin(
            [
                "future_test_top15_noncausal",
                "past_oof_rolling_120_q15",
                "past_oof_rolling_120_q20",
                "past_oof_rolling_120_q30",
                "nested_calib_fixed_q15",
                "nested_calib_rolling_120_q20",
                "nested_calib_rolling_120_q30",
                "nested_calib_rolling_120_q40",
            ]
        )
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
