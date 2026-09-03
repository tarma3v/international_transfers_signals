"""Deep train-block and labelled retrospective EDA for research round two.

No statistic from 2024--2026 is used to create a feature or choose a model.  The
period is nevertheless included in separate audit columns because it has already
been seen and is useful for diagnosing why older relationships fail.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from ml.targets import benefit_forward_only, build_targets
from research.extended_features import load_or_build

OUT = Path("results/research/round2")
BLOCKS = {
    "development": (2011, 2016),
    "general_validation": (2017, 2020),
    "transition": (2021, 2021),
    "shock_adaptation": (2022, 2023),
    "retrospective_final": (2024, 2026),
}


def _frame():
    X, names, index, series = load_or_build()
    targets = build_targets(series, index)
    frame = pd.DataFrame({
        "currency": [c for c, _i, _d in index],
        "series_i": [i for _c, i, _d in index],
        "date": [d for _c, _i, d in index],
        "year": [d.year for _c, _i, d in index],
        "month": [d.month for _c, _i, d in index],
        "fav_h1": targets["fav_h1"],
        "fav_h5": targets["fav_h5"],
    })
    for name in (
        "ret_1", "ret_5", "ret_20", "ret_60", "pct_range_90",
        "range_pos_20", "range_pos_60", "slope_z_20", "slope_z_60",
        "raw_vol_20", "raw_vol_60", "vol_ratio_20_120", "gap_days",
        "peer_ret_5_mean", "peer_dispersion_5", "usd_ret_5", "cny_ret_5",
    ):
        frame[name] = X[:, names.index(name)]
    benefit = np.full(len(index), np.nan)
    value = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value[row] = series[currency].values[i]
        b = benefit_forward_only(series[currency].values, i, 5)
        if b is not None:
            benefit[row] = b
    frame["value"] = value
    frame["benefit_h5"] = benefit
    return frame, X, names, index, series


def annual_panel(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (currency, year), z in frame.dropna(subset=["fav_h5"]).groupby(["currency", "year"]):
        rows.append({
            "currency": currency, "year": year, "n": len(z),
            "base_rate_h5": z.fav_h5.mean(),
            "mean_forward_benefit_bps": z.benefit_h5.mean(),
            "mean_ret1_bps": z.ret_1.mean(),
            "vol_ret1_bps": z.ret_1.std(),
            "mean_ret20_bps": z.ret_20.mean(),
            "median_gap_days": z.gap_days.median(),
            "share_gap_gt_1": (z.gap_days > 1).mean(),
        })
    return pd.DataFrame(rows)


def seasonal_stability(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    z = frame.dropna(subset=["fav_h5"])
    for block, (lo, hi) in BLOCKS.items():
        q = z[z.year.between(lo, hi)]
        for (currency, month), g in q.groupby(["currency", "month"]):
            rows.append({
                "block": block, "currency": currency, "month": month,
                "n": len(g), "base_rate_h5": g.fav_h5.mean(),
                "benefit_h5_bps": g.benefit_h5.mean(),
            })
    monthly = pd.DataFrame(rows)
    correlations = []
    for currency in CORRIDORS:
        p = monthly[monthly.currency == currency].pivot(
            index="month", columns="block", values="base_rate_h5"
        )
        blocks = list(BLOCKS)
        for i, left in enumerate(blocks):
            for right in blocks[i + 1:]:
                both = p[[left, right]].dropna()
                rho = spearmanr(both[left], both[right]).statistic if len(both) >= 4 else np.nan
                correlations.append({
                    "currency": currency, "left_block": left,
                    "right_block": right, "spearman_month_pattern": rho,
                    "months": len(both),
                })
    return monthly, pd.DataFrame(correlations)


def common_factor_and_leads(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = frame.pivot(index="date", columns="currency", values="ret_1").sort_index()
    pca_rows, lead_rows = [], []
    for block, (lo, hi) in BLOCKS.items():
        z = returns[[lo <= d.year <= hi for d in returns.index]].dropna()
        if len(z) < 30:
            continue
        scaled = StandardScaler().fit_transform(z)
        pca = PCA().fit(scaled)
        pca_rows.append({
            "block": block, "n_dates": len(z),
            "pc1_variance_share": pca.explained_variance_ratio_[0],
            "pc2_cumulative_share": pca.explained_variance_ratio_[:2].sum(),
            **{f"pc1_loading_{c}": pca.components_[0, list(z.columns).index(c)] for c in z.columns},
        })
        peer = z.mean(axis=1)
        for currency in z.columns:
            own = z[currency]
            other = (z.sum(axis=1) - own) / (len(z.columns) - 1)
            for lag in range(0, 6):
                lead_rows.append({
                    "block": block, "currency": currency, "peer_lag": lag,
                    "corr_own_with_lagged_peers": own.corr(other.shift(lag)),
                    "corr_own_with_common_factor": own.corr(peer.shift(lag)),
                })
    return pd.DataFrame(pca_rows), pd.DataFrame(lead_rows)


def offline_break_scan(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive only: full-block break candidates, never a causal feature."""
    rows = []
    for currency in CORRIDORS:
        z = frame[(frame.currency == currency) & frame.fav_h5.notna()].reset_index(drop=True)
        candidates = []
        for i in range(500, len(z) - 500, 20):
            left = z.iloc[max(0, i - 500):i]
            right = z.iloc[i:min(len(z), i + 500)]
            mean_scale = np.sqrt(left.ret_1.var() / len(left) + right.ret_1.var() / len(right) + 1e-9)
            mean_shift = abs(right.ret_1.mean() - left.ret_1.mean()) / mean_scale
            vol_shift = abs(np.log((right.ret_1.std() + 1e-6) / (left.ret_1.std() + 1e-6)))
            p = (left.fav_h5.sum() + right.fav_h5.sum()) / (len(left) + len(right))
            target_scale = np.sqrt(max(p * (1 - p), 1e-6) * (1 / len(left) + 1 / len(right)))
            target_shift = abs(right.fav_h5.mean() - left.fav_h5.mean()) / target_scale
            candidates.append({
                "currency": currency, "date": z.date.iloc[i],
                "mean_return_z": mean_shift, "log_vol_ratio_abs": vol_shift,
                "target_rate_z": target_shift,
                "composite": mean_shift + 2.0 * vol_shift + target_shift,
                "left_base": left.fav_h5.mean(), "right_base": right.fav_h5.mean(),
                "left_vol": left.ret_1.std(), "right_vol": right.ret_1.std(),
            })
        chosen = []
        for candidate in sorted(candidates, key=lambda r: r["composite"], reverse=True):
            if all(abs((candidate["date"] - old["date"]).days) >= 365 for old in chosen):
                chosen.append(candidate)
            if len(chosen) == 4:
                break
        rows.extend(chosen)
    return pd.DataFrame(rows).sort_values(["currency", "composite"], ascending=[True, False])


def observable_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed causal state definitions; results remain separated by data block."""
    z = frame.dropna(subset=["fav_h5"]).copy()
    z["trend_state"] = np.select(
        [z.ret_20 <= -100, z.ret_20 >= 100], ["down", "up"], default="flat"
    )
    z["vol_state"] = np.select(
        [z.vol_ratio_20_120 <= 0.70, z.vol_ratio_20_120 >= 1.35],
        ["quiet", "shock"], default="normal",
    )
    z["position_state"] = np.select(
        [z.range_pos_60 <= 25, z.range_pos_60 >= 75], ["low", "high"], default="middle"
    )
    rows = []
    for block, (lo, hi) in BLOCKS.items():
        q = z[z.year.between(lo, hi)]
        for dimension in ("trend_state", "vol_state", "position_state"):
            for state, g in q.groupby(dimension):
                rows.append({
                    "block": block, "dimension": dimension, "state": state,
                    "n": len(g), "base_rate_h5": g.fav_h5.mean(),
                    "benefit_h5_bps": g.benefit_h5.mean(),
                    "currencies": g.currency.nunique(),
                })
    return pd.DataFrame(rows)


def overlap_dependence(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for currency in CORRIDORS:
        y = frame.loc[frame.currency == currency, "fav_h5"]
        for lag in range(1, 11):
            rows.append({"currency": currency, "lag": lag, "label_autocorrelation": y.corr(y.shift(lag))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, _X, _names, _index, _series = _frame()
    annual_panel(frame).to_csv(OUT / "eda_annual_currency.csv", index=False)
    monthly, stability = seasonal_stability(frame)
    monthly.to_csv(OUT / "eda_monthly_blocks.csv", index=False)
    stability.to_csv(OUT / "eda_seasonal_stability.csv", index=False)
    pca, leads = common_factor_and_leads(frame)
    pca.to_csv(OUT / "eda_common_factor.csv", index=False)
    leads.to_csv(OUT / "eda_peer_lead_lag.csv", index=False)
    offline_break_scan(frame).to_csv(OUT / "eda_offline_break_candidates.csv", index=False)
    observable_regimes(frame).to_csv(OUT / "eda_observable_regimes.csv", index=False)
    overlap_dependence(frame).to_csv(OUT / "eda_label_overlap.csv", index=False)
    print("round-two EDA written to", OUT)


if __name__ == "__main__":
    main()

