"""Fetch a reproducible RUB/USD/CNY history from the official NBRB API."""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


BASE = "https://api.nbrb.by/exrates"
START = dt.date(2016, 1, 1)
END = dt.date(2026, 9, 5)
OUTPUT = Path("data/external_belarus_nbrb_rub_usd_cny_2016_2026.json")
CODES = ("RUB", "USD", "CNY")


def _get(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "itmo-fx-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    registry = _get(f"{BASE}/currencies")
    records = {code: {} for code in CODES}
    sources = []
    for item in registry:
        code = item["Cur_Abbreviation"]
        if code not in records:
            continue
        begin = max(START, dt.date.fromisoformat(item["Cur_DateStart"][:10]))
        finish = min(END, dt.date.fromisoformat(item["Cur_DateEnd"][:10]))
        if begin > finish:
            continue
        cursor = begin
        while cursor <= finish:
            chunk_end = min(finish, cursor + dt.timedelta(days=364))
            query = urllib.parse.urlencode({
                "startDate": cursor.isoformat(), "endDate": chunk_end.isoformat(),
            })
            url = f"{BASE}/rates/dynamics/{item['Cur_ID']}?{query}"
            rows = _get(url)
            for row in rows:
                day = row["Date"][:10]
                records[code][day] = float(row["Cur_OfficialRate"]) / float(item["Cur_Scale"])
            sources.append({
                "code": code, "cur_id": item["Cur_ID"],
                "start": cursor.isoformat(), "end": chunk_end.isoformat(),
                "n": len(rows), "url": url,
            })
            cursor = chunk_end + dt.timedelta(days=1)
            time.sleep(.05)
    payload = {
        "source": "National Bank of the Republic of Belarus official API",
        "retrieved_through": END.isoformat(),
        "sources": sources,
        "rates": {
            code: [[day, value] for day, value in sorted(mapping.items())]
            for code, mapping in records.items()
        },
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    for code, rows in payload["rates"].items():
        print(code, len(rows), rows[0][0], rows[-1][0])
    print(OUTPUT)


if __name__ == "__main__":
    main()
