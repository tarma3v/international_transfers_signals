"""Exploratory analysis restricted to the pre-2017 development sample."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

from ml.data import CORRIDORS
from ml.targets import HORIZONS, build_targets
from research.extended_features import load_or_build

TRAIN_END = dt.date(2016, 12, 31)
OUT = Path("results/research")


def _safe_lift(y: np.ndarray, mask: np.ndarray, scope: np.ndarray) -> float:
    base = float(np.nanmean(y[scope]))
    return float(np.nanmean(y[mask & scope]) / base) if np.any(mask & scope) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, index, series = load_or_build()
    dates = np.array([d for _c, _i, d in index], dtype=object)
    currencies = np.array([c for c, _i, _d in index], dtype=object)
    train = np.array([d <= TRAIN_END for d in dates])
    Y = build_targets(series, index)

    audit = []
    for code, s in series.items():
        gaps = np.array([(b - a).days for a, b in zip(s.dates[:-1], s.dates[1:])])
        calendar_span = (s.dates[-1] - s.dates[0]).days + 1
        audit.append({
            "currency": code,
            "rows": len(s),
            "first": s.dates[0],
            "last": s.dates[-1],
            "calendar_days": calendar_span,
            "missing_calendar_days": calendar_span - len(s),
            "share_nonpublication": 1.0 - len(s) / calendar_span,
            "max_gap_days": int(gaps.max()),
            "gaps_gt_4": int(np.sum(gaps > 4)),
            "nonpositive": int(np.sum(s.values <= 0)),
            "nan": int(np.sum(~np.isfinite(s.values))),
        })
    pd.DataFrame(audit).to_csv(OUT / "data_audit.csv", index=False)

    target_rows = []
    for h in HORIZONS:
        y = Y[f"fav_h{h}"]
        for code in CORRIDORS:
            m = train & (currencies == code) & ~np.isnan(y)
            target_rows.append({"h": h, "currency": code, "n": int(m.sum()),
                                "base_rate": float(y[m].mean())})
    pd.DataFrame(target_rows).to_csv(OUT / "train_target_rates.csv", index=False)

    # Rank features only inside development. Test and validation are untouched.
    y = Y["fav_h5"]
    usable = train & ~np.isnan(y)
    correlations = []
    for j, name in enumerate(names):
        if float(np.std(X[usable, j])) < 1e-12:
            continue
        corr = float(spearmanr(X[usable, j], y[usable]).statistic)
        correlations.append((name, corr, abs(corr)))
    correlations.sort(key=lambda row: -row[2])
    pd.DataFrame(correlations, columns=["feature", "spearman", "abs_spearman"]).to_csv(
        OUT / "train_feature_correlations.csv", index=False
    )

    # MI is more expensive, so evaluate the 160 strongest univariate candidates.
    candidate_names = [row[0] for row in correlations[:160]]
    candidate_cols = [names.index(name) for name in candidate_names]
    mi = mutual_info_classif(
        X[usable][:, candidate_cols], y[usable].astype(int), random_state=42,
        discrete_features=False,
    )
    mi_rows = sorted(zip(candidate_names, mi), key=lambda row: -row[1])
    pd.DataFrame(mi_rows, columns=["feature", "mutual_information"]).to_csv(
        OUT / "train_feature_mutual_information.csv", index=False
    )

    seasonal = []
    for code in CORRIDORS:
        cm = train & (currencies == code) & ~np.isnan(y)
        base = float(y[cm].mean())
        for field, values, levels in (
            ("month", np.array([d.month for d in dates]), range(1, 13)),
            ("dow", np.array([d.weekday() for d in dates]), range(7)),
        ):
            for level in levels:
                m = cm & (values == level)
                seasonal.append({"currency": code, "field": field, "level": level,
                                 "n": int(m.sum()), "hit_rate": float(y[m].mean()),
                                 "lift": float(y[m].mean() / base)})
    season_df = pd.DataFrame(seasonal)
    season_df.to_csv(OUT / "train_seasonality.csv", index=False)

    pct = X[:, names.index("pct_range_90")]
    bins = np.arange(0, 110, 10)
    bucket_rows = []
    for code in CORRIDORS:
        cm = usable & (currencies == code)
        base = float(y[cm].mean())
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = cm & (pct >= lo) & (pct < hi if hi < 100 else pct <= hi)
            bucket_rows.append({"currency": code, "lo": lo, "hi": hi, "n": int(m.sum()),
                                "hit_rate": float(y[m].mean()), "lift": float(y[m].mean()/base)})
    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(OUT / "train_pct_range_buckets.csv", index=False)

    # Visual summary uses development only.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    top = pd.DataFrame(correlations[:15], columns=["feature", "spearman", "abs"])
    axes[0, 0].barh(top.feature[::-1], top.spearman[::-1], color="#2563eb")
    axes[0, 0].set_title("Top Spearman associations with fav_h5 (train only)")
    axes[0, 0].axvline(0, color="black", linewidth=.7)

    for code in CORRIDORS:
        z = season_df[(season_df.currency == code) & (season_df.field == "month")]
        axes[0, 1].plot(z.level, z.lift, marker="o", label=code)
    axes[0, 1].axhline(1, color="black", linewidth=.7)
    axes[0, 1].set_title("Month lift by corridor (train only)")
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].legend(ncol=3, fontsize=8)

    for code in CORRIDORS:
        z = bucket_df[bucket_df.currency == code]
        axes[1, 0].plot((z.lo + z.hi) / 2, z.lift, marker="o", label=code)
    axes[1, 0].axhline(1, color="black", linewidth=.7)
    axes[1, 0].set_title("pct_range_90 buckets (train only)")
    axes[1, 0].set_xlabel("Range position")

    raw_ret = {}
    for code in CORRIDORS:
        keep = np.array([d <= TRAIN_END for d in series[code].dates])
        z = np.diff(np.log(series[code].values[keep])) * 10000
        raw_ret[code] = [_autocorr(z, lag) for lag in (1, 2, 5, 10, 20)]
    acf = pd.DataFrame(raw_ret, index=(1, 2, 5, 10, 20))
    acf.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Return autocorrelation (train only)")
    axes[1, 1].set_xlabel("Publication lag")
    axes[1, 1].axhline(0, color="black", linewidth=.7)
    fig.savefig(OUT / "train_only_eda.png", dpi=180)
    plt.close(fig)

    print(f"train rows: {int(usable.sum())}; features: {len(names)}")
    print("top train-only correlations:")
    for name, corr, _ in correlations[:20]:
        print(f"  {name:<30} {corr:+.4f}")
    print("top train-only mutual information:")
    for name, value in mi_rows[:15]:
        print(f"  {name:<30} {value:.4f}")
    print(f"artifacts: {OUT}")


def _autocorr(x: np.ndarray, lag: int) -> float:
    a, b = x[:-lag], x[lag:]
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else 0.0


if __name__ == "__main__":
    main()
