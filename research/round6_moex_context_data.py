"""Download the predeclared public MOEX risk/liquidity context packet."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FROM = "2010-01-01"
TILL = "2026-09-03"
DATA = Path("data/moex_market_context_2010_2026.json")
OUT = Path("results/research/round6/moex_context")
SPECS = {
    "IMOEX": {
        "url": "https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json",
        "columns": ("TRADEDATE", "SECID", "BOARDID", "OPEN", "HIGH", "LOW", "CLOSE", "VALUE", "YIELD"),
    },
    "RGBI": {
        "url": "https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/RGBI.json",
        "columns": ("TRADEDATE", "SECID", "BOARDID", "OPEN", "HIGH", "LOW", "CLOSE", "VALUE", "YIELD"),
    },
    "RUSFAR": {
        "url": "https://iss.moex.com/iss/history/engines/stock/markets/index/boards/MMIX/securities/RUSFAR.json",
        "columns": ("TRADEDATE", "SECID", "BOARDID", "OPEN", "HIGH", "LOW", "CLOSE", "VALUE", "YIELD"),
    },
    "GLDRUB_TOM": {
        "url": "https://iss.moex.com/iss/history/engines/currency/markets/selt/boards/CETS/securities/GLDRUB_TOM.json",
        "columns": ("TRADEDATE", "SECID", "BOARDID", "OPEN", "HIGH", "LOW", "CLOSE", "WAPRICE", "NUMTRADES"),
    },
}


def fetch_json(url):
    error = None
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "itmo-leakage-audit/1.0"})
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network retry
            error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"MOEX request failed: {url}") from error


def fetch_instrument(ticker, spec):
    rows, urls, start = [], [], 0
    while True:
        query = urlencode({
            "from": FROM, "till": TILL, "start": start,
            "iss.meta": "off", "iss.only": "history,history.cursor",
            "history.columns": ",".join(spec["columns"]),
        })
        url = f"{spec['url']}?{query}"
        payload = fetch_json(url)
        urls.append(url)
        block = payload["history"]
        if tuple(block["columns"]) != spec["columns"]:
            raise AssertionError(f"unexpected schema for {ticker}: {block['columns']}")
        rows.extend(block["data"])
        _index, total, page_size = map(int, payload["history.cursor"]["data"][0])
        start += page_size
        if start >= total:
            break
    dates = [row[0] for row in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise AssertionError(f"non-unique or unsorted dates for {ticker}")
    return {
        "ticker": ticker, "columns": spec["columns"], "rows": rows, "urls": urls,
    }


def main():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    instruments = [fetch_instrument(ticker, spec) for ticker, spec in SPECS.items()]
    payload = {
        "publisher": "Moscow Exchange", "api": "MOEX ISS public history",
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
    print(json.dumps({**manifest, "source_urls": f"{len(manifest['source_urls'])} URLs"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
