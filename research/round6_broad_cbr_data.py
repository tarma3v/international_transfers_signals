"""Build a target-free broad FX reference panel from official CBR XML.

The five prediction corridors are deliberately excluded.  Currency IDs are
resolved from the CBR directory instead of copied from an unofficial table,
and every observation is divided by its row-specific nominal.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


# The ``Full`` variant is the official directory endpoint that includes ISO
# character codes; XML_val.asp intentionally omits them.
DIRECTORY_URL = "https://www.cbr.ru/scripts/XML_valFull.asp?d=0"
DYNAMIC_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
TARGETS = {"AMD", "KGS", "KZT", "TJS", "UZS"}
# Frozen before looking at target labels: major/liquid currencies plus diverse
# regional references.  Availability coverage, not predictive performance,
# decides which of these survives into the panel.
REQUESTED_ISO = (
    "AUD", "BRL", "BYN", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR",
    "GBP", "HKD", "HUF", "INR", "JPY", "KRW", "MXN", "NOK", "NZD",
    "PLN", "SEK", "SGD", "TRY", "USD", "ZAR",
)
USER_AGENT = "international-transfers-signals/round6 research"


def _get(url: str, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"CBR did not answer after {retries} attempts: {url}") from last


def parse_directory(raw: bytes) -> dict[str, list[dict[str, str]]]:
    root = ET.fromstring(raw.decode("windows-1251"))
    found: dict[str, list[dict[str, str]]] = {}
    for item in root.findall("Item"):
        iso_node = item.find("ISO_Char_Code")
        if iso_node is None or not (iso_node.text or "").strip():
            continue
        iso = (iso_node.text or "").strip().upper()
        found.setdefault(iso, []).append({
            "id": (item.get("ID") or "").strip(),
            "name": (item.findtext("Name") or "").strip(),
            "english_name": (item.findtext("EngName") or "").strip(),
            "parent_code": (item.findtext("ParentCode") or "").strip(),
        })
    return found


def fetch_history(currency_id: str, date_from: str, date_to: str) -> tuple[str, list[dict]]:
    query = urllib.parse.urlencode({
        "date_req1": date_from,
        "date_req2": date_to,
        "VAL_NM_RQ": currency_id,
    })
    url = f"{DYNAMIC_URL}?{query}"
    raw = _get(url)
    root = ET.fromstring(raw.decode("windows-1251"))
    rows = []
    for record in root.findall("Record"):
        day = dt.datetime.strptime(record.get("Date") or "", "%d.%m.%Y").date()
        nominal = float((record.findtext("Nominal") or "nan").replace(",", "."))
        value = float((record.findtext("Value") or "nan").replace(",", "."))
        if nominal <= 0:
            raise ValueError(f"invalid nominal for {currency_id} on {day}: {nominal}")
        rows.append({
            "date": day.isoformat(),
            "rub_per_unit": value / nominal,
            "nominal": nominal,
        })
    rows.sort(key=lambda row: row["date"])
    return url, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", default="01/01/2010")
    parser.add_argument("--to", dest="date_to", default="05/09/2026")
    parser.add_argument(
        "--out", default="data/cbr_broad_reference_2010_2026.json",
    )
    parser.add_argument(
        "--manifest",
        default="results/research/round6/broad_cbr/data_manifest.json",
    )
    parser.add_argument("--minimum-usd-coverage", type=float, default=.65)
    args = parser.parse_args()

    raw_directory = _get(DIRECTORY_URL)
    directory = parse_directory(raw_directory)
    missing = sorted(set(REQUESTED_ISO) - set(directory))
    if TARGETS.intersection(REQUESTED_ISO):
        raise AssertionError("target corridor entered broad reference list")

    fetched: dict[str, dict] = {}
    alternatives: dict[str, list[dict]] = {}
    for iso in (code for code in REQUESTED_ISO if code in directory):
        candidates = directory[iso]
        alternatives[iso] = candidates
        candidate_histories = []
        for candidate in candidates:
            url, rows = fetch_history(candidate["id"], args.date_from, args.date_to)
            candidate_histories.append((len(rows), candidate, url, rows))
        _count, chosen, url, rows = max(candidate_histories, key=lambda item: item[0])
        if not rows:
            continue
        fetched[iso] = {
            "currency_id": chosen["id"],
            "name": chosen["name"],
            "source_url": url,
            "rows": rows,
        }
        print(f"{iso}: {len(rows)} rows, {rows[0]['date']} -> {rows[-1]['date']}, "
              f"nominals={sorted({row['nominal'] for row in rows})}")

    if "USD" not in fetched:
        raise RuntimeError("USD history is required as the coverage reference")
    usd_rows = len(fetched["USD"]["rows"])
    retained = {
        iso: item for iso, item in fetched.items()
        if len(item["rows"]) / usd_rows >= args.minimum_usd_coverage
    }
    rejected = {
        iso: len(item["rows"]) / usd_rows for iso, item in fetched.items()
        if iso not in retained
    }
    if len(retained) < 12:
        raise RuntimeError(f"broad panel unexpectedly small: {sorted(retained)}")

    payload = {
        "schema": 1,
        "source": "Bank of Russia official XML API",
        "directory_url": DIRECTORY_URL,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "targets_excluded": sorted(TARGETS),
        "selection_rule": {
            "requested_iso": list(REQUESTED_ISO),
            "minimum_usd_row_coverage": args.minimum_usd_coverage,
        },
        "currencies": retained,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)

    manifest = {
        "schema": 1,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "directory_url": DIRECTORY_URL,
        "directory_sha256": hashlib.sha256(raw_directory).hexdigest(),
        "payload_path": str(output_path),
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "usd_rows": usd_rows,
        "retained": {
            iso: {
                "currency_id": item["currency_id"],
                "rows": len(item["rows"]),
                "coverage_vs_usd": len(item["rows"]) / usd_rows,
                "first_date": item["rows"][0]["date"],
                "last_date": item["rows"][-1]["date"],
            }
            for iso, item in retained.items()
        },
        "rejected_for_coverage": rejected,
        "absent_from_daily_directory": missing,
        "directory_alternatives": alternatives,
        "target_labels_used_for_selection": False,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"retained={len(retained)} rejected={rejected}")
    print(f"payload={output_path} sha256={manifest['payload_sha256']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
