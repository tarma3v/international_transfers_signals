"""Download the frozen official MOEX hourly perpetual-FX candle packet."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

from research.round6_moex_data import fetch_json


TICKERS = ("CNYRUBF", "USDRUBF")
FROM = "2022-01-01"
TILL = "2026-09-03"
INTERVAL = 60
PAGE_SIZE = 500
COLUMNS = ("open", "close", "high", "low", "value", "volume", "begin", "end")
BASE = (
    "https://iss.moex.com/iss/engines/futures/markets/forts/"
    "securities/{ticker}/candles.json"
)
DATA = Path("data/moex_perpetual_fx_hourly_2022_2026.json")
OUT = Path("results/research/round6/moex_perpetual_hourly")


def fetch_ticker(ticker: str) -> dict:
    rows, urls, start = [], [], 0
    while True:
        query = urlencode({
            "from": FROM,
            "till": TILL,
            "interval": INTERVAL,
            "start": start,
            "iss.meta": "off",
            "iss.only": "candles",
        })
        url = f"{BASE.format(ticker=ticker)}?{query}"
        payload = fetch_json(url)
        urls.append(url)
        block = payload["candles"]
        if tuple(block["columns"]) != COLUMNS:
            raise AssertionError(f"unexpected hourly schema: {block['columns']}")
        page = block["data"]
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += len(page)
    begin = [row[COLUMNS.index("begin")] for row in rows]
    if begin != sorted(begin) or len(begin) != len(set(begin)):
        raise AssertionError(f"non-unique or unsorted hourly candles for {ticker}")
    for row in rows:
        if any(row[COLUMNS.index(name)] is None or float(row[COLUMNS.index(name)]) <= 0
               for name in ("open", "close", "high", "low")):
            raise AssertionError(f"invalid OHLC candle for {ticker}: {row}")
    return {"ticker": ticker, "columns": COLUMNS, "rows": rows, "urls": urls}


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(TICKERS)) as executor:
        loaded = dict(zip(TICKERS, executor.map(fetch_ticker, TICKERS)))
    instruments = [loaded[ticker] for ticker in TICKERS]
    payload = {
        "publisher": "Moscow Exchange",
        "api": "MOEX ISS public 60-minute futures candles",
        "timezone": "Europe/Moscow",
        "from": FROM,
        "till": TILL,
        "interval_minutes": INTERVAL,
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "instruments": instruments,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    DATA.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    manifest = {
        "file": str(DATA),
        "payload_sha256": digest,
        "decision_time": "12:00:00 Europe/Moscow",
        "strict_asof_rule": "candle end < signal_date 12:00:00",
        "next_cbr_fixing_loaded": False,
        "instruments": [{
            "ticker": item["ticker"],
            "rows": len(item["rows"]),
            "first_begin": item["rows"][0][-2] if item["rows"] else None,
            "last_end": item["rows"][-1][-1] if item["rows"] else None,
            "pages": len(item["urls"]),
        } for item in instruments],
        "source_urls": [url for item in instruments for url in item["urls"]],
    }
    (OUT / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(
        {**manifest, "source_urls": f"{len(manifest['source_urls'])} URLs"},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
