"""Build conservatively release-lagged external features for round two.

The joins are by *availability date*, never by observation date.  RUONIA has an
explicit publication timestamp in the CBR workbook.  The FRED/EIA series use
conservative fixed lags and are evaluated at multiple lags because the downloaded
latest-vintage CSV does not itself preserve every historical release timestamp.
Consequently external-data results remain sensitivity analysis, not the headline.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("results/research/round2")
BRENT = DATA / "external_dcoilbrenteu.csv"
DOLLAR = DATA / "external_dtwexbgs.csv"
RUONIA = DATA / "external_ruonia.xlsx"
KEYRATE = DATA / "external_keyrate.html"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _market_csv(path: Path, value_name: str, lag_days: int) -> pd.DataFrame:
    z = pd.read_csv(path, na_values=".")
    z.columns = ["observation_date", value_name]
    z["observation_date"] = pd.to_datetime(z.observation_date)
    z[value_name] = pd.to_numeric(z[value_name], errors="coerce")
    z = z.dropna().sort_values("observation_date")
    z["available_date"] = z.observation_date + pd.to_timedelta(lag_days, unit="D")
    return z[["available_date", value_name]]


def _ruonia() -> pd.DataFrame:
    z = pd.read_excel(RUONIA)
    z["available_date"] = pd.to_datetime(z["DateUpdate"]).dt.normalize()
    rename = {
        "ruo": "ruonia", "vol": "ruonia_volume", "T": "ruonia_trades",
        "C": "ruonia_participants", "MinRate": "ruonia_min",
        "Percentile25": "ruonia_p25", "Percentile75": "ruonia_p75",
        "MaxRate": "ruonia_max",
    }
    z = z.rename(columns=rename).sort_values("available_date")
    cols = ["available_date"] + list(rename.values())
    return z[cols].drop_duplicates("available_date", keep="last")


def _keyrate() -> pd.DataFrame:
    text = KEYRATE.read_text(encoding="utf-8")
    pairs = re.findall(
        r"<tr>\s*<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td>([\d,]+)</td>\s*</tr>",
        text,
    )
    if not pairs:
        raise ValueError("key-rate rows not found")
    z = pd.DataFrame(pairs, columns=["available_date", "key_rate"])
    z["available_date"] = pd.to_datetime(z.available_date, dayfirst=True)
    z["key_rate"] = z.key_rate.str.replace(",", ".", regex=False).astype(float)
    return z.sort_values("available_date").drop_duplicates("available_date", keep="last")


def _asof(calendar: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    source = source.dropna(subset=["available_date"])
    return pd.merge_asof(
        calendar.sort_values("date"), source.sort_values("available_date"),
        left_on="date", right_on="available_date", direction="backward",
    ).drop(columns="available_date")


def _technical(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    x = frame[column].astype(float).ffill()
    for lag in (1, 5, 20, 60):
        frame[f"{column}_ret_{lag}"] = np.log(x / x.shift(lag)).replace([np.inf, -np.inf], np.nan) * 10000
    frame[f"{column}_vol_20"] = frame[f"{column}_ret_1"].rolling(20, min_periods=5).std()
    lo = x.rolling(60, min_periods=10).min(); hi = x.rolling(60, min_periods=10).max()
    frame[f"{column}_range_pos_60"] = (x - lo) / (hi - lo).replace(0, np.nan) * 100
    return frame


def build(lag_brent: int, lag_dollar: int) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": pd.date_range("2010-01-01", "2026-09-03", freq="D")})
    z = _asof(calendar, _ruonia())
    z = _asof(z, _keyrate())
    z = _asof(z, _market_csv(BRENT, "brent", lag_brent))
    z = _asof(z, _market_csv(DOLLAR, "broad_dollar", lag_dollar))
    for col in ("brent", "broad_dollar"):
        z = _technical(z, col)
    for col in ("ruonia", "ruonia_volume", "key_rate"):
        x = z[col].astype(float).ffill()
        z[f"{col}_change_5"] = x - x.shift(5)
        z[f"{col}_change_20"] = x - x.shift(20)
    z["ruonia_key_spread"] = z.ruonia - z.key_rate
    z["ruonia_intraday_range"] = z.ruonia_max - z.ruonia_min
    z["days_since_key_change"] = 0.0
    last = None
    previous = None
    for row in z.index:
        value = z.at[row, "key_rate"]
        if pd.notna(value) and (previous is None or value != previous):
            last = z.at[row, "date"]
        z.at[row, "days_since_key_change"] = (
            (z.at[row, "date"] - last).days if last is not None else np.nan
        )
        if pd.notna(value): previous = value
    keep = [c for c in z.columns if c not in ("brent", "broad_dollar")]
    return z[keep]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = []
    for lag_brent, lag_dollar in ((2, 7), (5, 10), (7, 14)):
        name = f"external_features_b{lag_brent}_d{lag_dollar}.csv"
        z = build(lag_brent, lag_dollar)
        z.to_csv(OUT / name, index=False)
        manifests.append({
            "file": name, "brent_availability_lag_calendar_days": lag_brent,
            "broad_dollar_availability_lag_calendar_days": lag_dollar,
            "rows": len(z), "first": str(z.date.min().date()),
            "last": str(z.date.max().date()),
        })
    provenance = {
        "sources": [
            {"file": str(BRENT), "sha256": _sha(BRENT),
             "url": "https://fred.stlouisfed.org/series/DCOILBRENTEU",
             "publisher": "U.S. Energy Information Administration via FRED"},
            {"file": str(DOLLAR), "sha256": _sha(DOLLAR),
             "url": "https://fred.stlouisfed.org/series/DTWEXBGS",
             "publisher": "Federal Reserve Board via FRED"},
            {"file": str(RUONIA), "sha256": _sha(RUONIA),
             "url": "https://www.cbr.ru/hd_base/ruonia/dynamics/",
             "publisher": "Bank of Russia", "availability": "explicit DateUpdate"},
            {"file": str(KEYRATE), "sha256": _sha(KEYRATE),
             "url": "https://www.cbr.ru/hd_base/keyrate/",
             "publisher": "Bank of Russia", "availability": "effective-date table"},
        ],
        "sensitivity_files": manifests,
        "limitation": (
            "FRED latest-vintage CSVs lack row-level historical release timestamps; "
            "fixed conservative lags are sensitivity assumptions, not exact vintages."
        ),
    }
    (OUT / "external_data_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
