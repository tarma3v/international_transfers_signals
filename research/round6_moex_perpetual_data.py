"""Download the predeclared public MOEX perpetual FX futures packet."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

from research.round6_moex_data import fetch_json


TICKERS = ("CNYRUBF", "USDRUBF")
FROM = "2022-01-01"
TILL = "2026-09-03"
COLUMNS = (
    "TRADEDATE", "SECID", "OPEN", "LOW", "HIGH", "CLOSE",
    "OPENPOSITIONVALUE", "VALUE", "VOLUME", "OPENPOSITION",
    "SETTLEPRICE", "SWAPRATE", "WAPRICE", "CHANGE", "QTY",
    "NUMTRADES", "ASSETCODE",
)
BASE = (
    "https://iss.moex.com/iss/history/engines/futures/markets/forts/"
    "securities/{ticker}.json"
)
DATA = Path("data/moex_perpetual_fx_history_2022_2026.json")
OUT = Path("results/research/round6/moex_perpetual")


def fetch_ticker(ticker):
    rows, urls, start = [], [], 0
    while True:
        query = urlencode({
            "from": FROM,
            "till": TILL,
            "start": start,
            "iss.meta": "off",
            "iss.only": "history,history.cursor",
            "history.columns": ",".join(COLUMNS),
        })
        url = f"{BASE.format(ticker=ticker)}?{query}"
        payload = fetch_json(url)
        urls.append(url)
        block = payload["history"]
        if tuple(block["columns"]) != COLUMNS:
            raise AssertionError(f"unexpected MOEX futures schema: {block['columns']}")
        rows.extend(block["data"])
        _index, total, page_size = map(
            int, payload["history.cursor"]["data"][0],
        )
        start += page_size
        if start >= total:
            break
    dates = [row[0] for row in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise AssertionError(f"non-unique or unsorted futures dates for {ticker}")
    return {"ticker": ticker, "columns": COLUMNS, "rows": rows, "urls": urls}


def main():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    instruments = [fetch_ticker(ticker) for ticker in TICKERS]
    payload = {
        "publisher": "Moscow Exchange",
        "api": "MOEX ISS public futures history",
        "from": FROM,
        "till": TILL,
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
        "strict_asof_rule": "TRADEDATE < signal_date",
        "same_day_futures_values_allowed": False,
        "instruments": [{
            "ticker": item["ticker"],
            "rows": len(item["rows"]),
            "first": item["rows"][0][0] if item["rows"] else None,
            "last": item["rows"][-1][0] if item["rows"] else None,
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
