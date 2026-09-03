# RECONNAISSANCE ONLY (day-1 context gathering) - not product code
import urllib.request, xml.etree.ElementTree as ET, datetime as dt, statistics, json, collections

IDS = {"TJS":"R01670","UZS":"R01717","KGS":"R01370","AMD":"R01060","KZT":"R01335","USD":"R01235","CNY":"R01375"}
d1, d2 = "01/01/2019", "02/09/2026"
out = {}
for code, vid in IDS.items():
    url = f"https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={d1}&date_req2={d2}&VAL_NM_RQ={vid}"
    raw = urllib.request.urlopen(url, timeout=60).read().decode("windows-1251")
    root = ET.fromstring(raw)
    rows = []
    for r in root.findall("Record"):
        date = dt.datetime.strptime(r.get("Date"), "%d.%m.%Y").date()
        nom = float(r.find("Nominal").text.replace(",", "."))
        val = float(r.find("Value").text.replace(",", "."))
        rows.append((date, val/nom, nom))   # RUB per 1 unit of foreign currency
    rows.sort()
    out[code] = rows
    noms = sorted(set(n for _,_,n in rows))
    print(f"{code}: {len(rows)} rows, {rows[0][0]} -> {rows[-1][0]}, nominals seen={noms}, last={rows[-1][1]:.6f} RUB")
json.dump({k: [(str(d), v) for d, v, _ in r] for k, r in out.items()}, open("data/cbr_rates.json", "w"))
