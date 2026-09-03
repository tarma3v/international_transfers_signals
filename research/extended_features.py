"""Extended causal features for the 2010--2026 CBR publication-time panel.

Every numeric feature is computed from observations whose publication date is
not later than the row date.  The public ``causality_check`` rebuilds the matrix
after truncating the future and requires exact equality on the retained rows.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.data import CORRIDORS, Series, load
from ml.features import build_matrix

LONG_DATA = Path("data/cbr_rates_2010_2026.json")
REFERENCES = ("USD", "CNY", "EUR")
CACHE = Path("research/cache/extended_features_2010_2026.npz")


def _position(x: np.ndarray) -> float:
    lo, hi = float(np.min(x)), float(np.max(x))
    return 50.0 if hi <= lo else (float(x[-1]) - lo) / (hi - lo) * 100.0


def _percentile(x: np.ndarray) -> float:
    return 50.0 if len(x) < 2 else float(np.mean(x[:-1] <= x[-1]) * 100.0)


def _log_return(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return 0.0
    return float(np.log(x[-1] / x[-1 - lag]) * 10000.0)


def _slope_z(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    z = np.log(x)
    t = np.arange(len(z), dtype=float)
    slope = float(np.polyfit(t, z, 1)[0])
    resid = z - (slope * t + float(z.mean() - slope * t.mean()))
    scale = float(resid.std())
    return slope * len(z) / scale if scale > 1e-12 else 0.0


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 3 or float(np.std(x)) < 1e-12:
        return 0.0
    a, b = x[:-lag], x[lag:]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rolling_beta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 10 or len(b) != len(a):
        return 0.0, 0.0
    vb = float(np.var(b))
    beta = float(np.cov(a, b, ddof=0)[0, 1] / vb) if vb > 1e-12 else 0.0
    if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = 0.0
    return beta, corr


def _aligned_past(series: dict[str, Series], code: str, day: dt.date) -> np.ndarray:
    dates = series[code].dates
    j = int(np.searchsorted(dates, day, side="right"))
    return series[code].values[:j]


def _extra_row(series: dict[str, Series], currency: str, i: int, day: dt.date) -> dict[str, float]:
    own = series[currency].values[: i + 1]
    f: dict[str, float] = {}

    for lag in (2, 4, 7, 15, 30, 45, 90, 180, 500):
        f[f"raw_ret_{lag}"] = _log_return(own, lag)
    one_step = np.diff(np.log(own[-251:])) * 10000.0
    for lag in range(1, 11):
        f[f"raw_ret1_lag_{lag}"] = float(one_step[-lag]) if len(one_step) >= lag else 0.0

    for window in (10, 20, 60, 120, 250, 500):
        w = own[-window:]
        f[f"range_pos_{window}"] = _position(w)
        f[f"rank_level_{window}"] = _percentile(w)
        f[f"slope_z_{window}"] = _slope_z(w)
        f[f"new_high_{window}"] = float(len(w) > 1 and w[-1] >= np.max(w[:-1]))
        f[f"new_low_{window}"] = float(len(w) > 1 and w[-1] <= np.min(w[:-1]))
        mean = float(np.mean(w))
        f[f"level_vs_mean_{window}"] = (float(w[-1]) / mean - 1.0) * 10000.0

    for span in (5, 10, 20, 60, 120):
        w = own[-min(len(own), span * 5):]
        alpha = 2.0 / (span + 1.0)
        ema = float(w[0])
        for value in w[1:]:
            ema = alpha * float(value) + (1.0 - alpha) * ema
        f[f"level_vs_ema_{span}"] = (float(own[-1]) / ema - 1.0) * 10000.0
    f["macd_5_20"] = f["level_vs_ema_20"] - f["level_vs_ema_5"]
    f["macd_20_60"] = f["level_vs_ema_60"] - f["level_vs_ema_20"]

    for window in (5, 20, 60, 120, 250):
        r = one_step[-window:]
        if len(r) == 0:
            r = np.array([0.0])
        med = float(np.median(r))
        sd = float(np.std(r))
        f[f"raw_vol_{window}"] = sd
        f[f"ret_median_{window}"] = med
        f[f"ret_mad_{window}"] = float(np.median(np.abs(r - med)))
        f[f"positive_share_{window}"] = float(np.mean(r > 0))
        f[f"max_up_{window}"] = float(np.max(r))
        f[f"max_down_{window}"] = float(np.min(r))
        f[f"ret_ac1_{window}"] = _autocorr(r, 1)
        f[f"ret_ac5_{window}"] = _autocorr(r, 5)
        if sd > 1e-12:
            z = (r - float(np.mean(r))) / sd
            f[f"ret_kurt_{window}"] = float(np.mean(z**4) - 3.0)
        else:
            f[f"ret_kurt_{window}"] = 0.0
    f["vol_ratio_5_60"] = f["raw_vol_5"] / max(f["raw_vol_60"], 1e-9)
    f["vol_ratio_20_120"] = f["raw_vol_20"] / max(f["raw_vol_120"], 1e-9)
    f["high_vol_regime"] = float(f["vol_ratio_20_120"] >= 1.35)
    f["quiet_regime"] = float(f["vol_ratio_20_120"] <= 0.70)

    # Raw and binary calendar encodings coexist: linear and tree models need
    # different representations, and the choice is made on validation only.
    f["calendar_dow"] = float(day.weekday())
    f["calendar_dom"] = float(day.day)
    f["calendar_month"] = float(day.month)
    f["calendar_doy"] = float(day.timetuple().tm_yday)
    f["calendar_week"] = float(day.isocalendar()[1])
    for m in range(1, 13):
        f[f"is_month_{m:02d}"] = float(day.month == m)
    for dow in range(7):
        f[f"is_dow_{dow}"] = float(day.weekday() == dow)
    year_days = 366.0 if day.year % 4 == 0 else 365.0
    for harmonic in range(1, 5):
        angle = 2.0 * np.pi * harmonic * (day.timetuple().tm_yday - 1) / year_days
        f[f"annual_sin_{harmonic}"] = float(np.sin(angle))
        f[f"annual_cos_{harmonic}"] = float(np.cos(angle))
    f["first_week_month"] = float(day.day <= 7)
    f["last_week_month"] = float(day.day >= 24)
    f["pre_new_year_7"] = float(day.month == 12 and day.day >= 24)
    f["pre_new_year_14"] = float(day.month == 12 and day.day >= 17)
    f["post_new_year_7"] = float(day.month == 1 and day.day <= 7)
    f["pre_sep_first_7"] = float(day.month == 8 and day.day >= 25)
    f["pre_sep_first_14"] = float(day.month == 8 and day.day >= 18)
    f["summer"] = float(day.month in (6, 7, 8))
    f["year_end"] = float(day.month in (11, 12))

    own_r = np.diff(np.log(own[-121:])) * 10000.0
    reference_returns: dict[str, np.ndarray] = {}
    for ref in REFERENCES:
        ref_values = _aligned_past(series, ref, day)
        for lag in (1, 2, 5, 10, 20, 60, 120):
            f[f"{ref.lower()}_raw_ret_{lag}"] = _log_return(ref_values, lag)
        ref_r = np.diff(np.log(ref_values[-121:])) * 10000.0
        n = min(len(own_r), len(ref_r), 120)
        if n >= 10:
            beta, corr = _rolling_beta(own_r[-n:], ref_r[-n:])
        else:
            beta, corr = 0.0, 0.0
        f[f"beta_{ref.lower()}_120"] = beta
        f[f"corr_{ref.lower()}_120"] = corr
        reference_returns[ref] = ref_r

    # Cross-rates remove the common RUB leg and expose global USD/EUR/CNY moves.
    aligned_refs = {r: _aligned_past(series, r, day) for r in REFERENCES}
    nref = min(map(len, aligned_refs.values()))
    if nref > 60:
        usd, cny, eur = (aligned_refs[k][-nref:] for k in ("USD", "CNY", "EUR"))
        for lag in (5, 20, 60):
            f[f"eurusd_ret_{lag}"] = _log_return(eur / usd, lag)
            f[f"cnyusd_ret_{lag}"] = _log_return(cny / usd, lag)
    else:
        for lag in (5, 20, 60):
            f[f"eurusd_ret_{lag}"] = f[f"cnyusd_ret_{lag}"] = 0.0
    return f


def build_extended_matrix(series: dict[str, Series]) -> tuple[np.ndarray, list[str], list[tuple]]:
    base, base_names, index = build_matrix(series, CORRIDORS, REFERENCES)
    extras = [_extra_row(series, c, i, day) for c, i, day in index]
    extra_names = sorted(extras[0])
    extra_matrix = np.asarray([[row[name] for name in extra_names] for row in extras], dtype=float)
    X = np.column_stack([base, extra_matrix])
    if not np.all(np.isfinite(X)):
        bad = np.where(~np.isfinite(X))
        raise ValueError(f"non-finite extended features at {bad[0][0]}, {bad[1][0]}")
    return X, base_names + extra_names, index


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_or_build(rebuild: bool = False):
    fingerprint = _fingerprint(LONG_DATA)
    if CACHE.exists() and not rebuild:
        cached = np.load(CACHE, allow_pickle=True)
        if str(cached["fingerprint"].item()) == fingerprint:
            index = [(str(c), int(i), d) for c, i, d in cached["index"]]
            return cached["X"], list(cached["names"]), index, load(LONG_DATA)
    series = load(LONG_DATA)
    X, names, index = build_extended_matrix(series)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        X=X,
        names=np.asarray(names, dtype=object),
        index=np.asarray(index, dtype=object),
        fingerprint=np.asarray(fingerprint, dtype=object),
    )
    return X, names, index, series


def causality_check(cut: dt.date = dt.date(2020, 12, 31)) -> None:
    """Rebuild on physically truncated inputs and compare every retained row."""
    full = load(LONG_DATA)
    X_full, names_full, idx_full = build_extended_matrix(full)
    truncated = {}
    for code, s in full.items():
        keep = np.array([d <= cut for d in s.dates])
        truncated[code] = Series(code, s.dates[keep].copy(), s.values[keep].copy())
    X_cut, names_cut, idx_cut = build_extended_matrix(truncated)
    if names_full != names_cut:
        raise AssertionError("feature names changed after truncation")
    expected = [row for row in idx_full if row[2] <= cut]
    if expected != idx_cut:
        raise AssertionError("row index changed before the cut")
    full_rows = {row: r for r, row in enumerate(idx_full)}
    reference = X_full[[full_rows[row] for row in idx_cut]]
    if not np.allclose(reference, X_cut, rtol=1e-12, atol=1e-12):
        diff = np.where(~np.isclose(reference, X_cut, rtol=1e-12, atol=1e-12))
        raise AssertionError(f"future leakage in {names_full[int(diff[1][0])]}")


if __name__ == "__main__":
    X, names, index, _series = load_or_build(rebuild=True)
    print(json.dumps({"rows": len(X), "features": len(names), "first": str(index[0][2]),
                      "last": str(index[-1][2])}, ensure_ascii=False))
