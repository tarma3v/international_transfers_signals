"""Download and archive the predeclared public MOEX FX history packet."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TICKERS = ("CNYRUB_TOM", "USD000UTSTOM", "EUR_RUB__TOM")
FROM = "2010-01-01"
TILL = "2026-09-03"
COLUMNS = ("TRADEDATE", "SECID", "BOARDID", "OPEN", "HIGH", "LOW", "CLOSE", "WAPRICE", "NUMTRADES")
BASE = "https://iss.moex.com/iss/history/engines/currency/markets/selt/boards/CETS/securities/{ticker}.json"
DATA = Path("data/moex_fx_history_2010_2026.json")
OUT = Path("results/research/round6/moex")


def fetch_json(url: str):
    error = None
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "itmo-leakage-audit/1.0"})
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network retry
            error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"MOEX request failed: {url}") from error


def fetch_ticker(ticker: str):
    rows, start, urls = [], 0, []
    while True:
        query = urlencode({
            "from": FROM, "till": TILL, "start": start,
            "iss.meta": "off", "iss.only": "history,history.cursor",
            "history.columns": ",".join(COLUMNS),
        })
        url = f"{BASE.format(ticker=ticker)}?{query}"
        payload = fetch_json(url)
        urls.append(url)
        block = payload["history"]
        if tuple(block["columns"]) != COLUMNS:
            raise AssertionError(f"unexpected MOEX schema for {ticker}: {block['columns']}")
        rows.extend(block["data"])
        cursor = payload["history.cursor"]["data"][0]
        _index, total, page_size = map(int, cursor)
        start += page_size
        if start >= total:
            break
    dates = [row[0] for row in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise AssertionError(f"non-unique or unsorted MOEX dates for {ticker}")
    return {"ticker": ticker, "columns": COLUMNS, "rows": rows, "urls": urls}


def main():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    instruments = [fetch_ticker(ticker) for ticker in TICKERS]
    payload = {
        "publisher": "Moscow Exchange",
        "api": "MOEX ISS public history",
        "from": FROM, "till": TILL,
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "instruments": instruments,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    DATA.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    manifest = {
        "file": str(DATA), "payload_sha256": digest,
        "strict_asof_rule": "TRADEDATE < signal_date",
        "same_day_close_allowed": False,
        "instruments": [{
            "ticker": item["ticker"], "rows": len(item["rows"]),
            "first": item["rows"][0][0] if item["rows"] else None,
            "last": item["rows"][-1][0] if item["rows"] else None,
            "pages": len(item["urls"]),
        } for item in instruments],
        "source_urls": [url for item in instruments for url in item["urls"]],
    }
    (OUT / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
