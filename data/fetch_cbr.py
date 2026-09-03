"""Выгрузка официальных курсов ЦБ РФ — единственный источник данных проекта.

Ключевой шаг здесь — построчная нормировка на номинал (value/nominal). Номинал ЦБ
менялся в середине истории (TJS 1->10, UZS 1000->10000, KGS 10->100, CNY 1->10),
и постоянный делитель дал бы разрыв курса в 10 раз, который любая модель прочитает
как настоящее движение рынка. Кейс отдельно предупреждает об этой ловушке.

Запуск: python data/fetch_cbr.py [--from DD/MM/YYYY] [--to DD/MM/YYYY] [--out PATH]
Печатает число строк и SHA-256 выгрузки — их можно сверить с results/.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import urllib.request
import xml.etree.ElementTree as ET

IDS = {"TJS": "R01670", "UZS": "R01717", "KGS": "R01370", "AMD": "R01060",
       "KZT": "R01335", "USD": "R01235", "CNY": "R01375", "EUR": "R01239"}
MIN_ROWS = 1000  # меньше — значит ЦБ вернул усечённый ответ, дальше идти нельзя


def fetch(code: str, vid: str, d1: str, d2: str, retries: int = 3) -> list:
    url = (f"https://www.cbr.ru/scripts/XML_dynamic.asp"
           f"?date_req1={d1}&date_req2={d2}&VAL_NM_RQ={vid}")
    last: Exception | None = None
    for _ in range(retries):
        try:
            raw = urllib.request.urlopen(url, timeout=30).read().decode("windows-1251")
        except Exception as exc:  # сеть: пробуем ещё раз, но не молча
            last = exc
            continue
        rows = []
        for r in ET.fromstring(raw).findall("Record"):
            date = dt.datetime.strptime(r.get("Date"), "%d.%m.%Y").date()
            nom = float(r.find("Nominal").text.replace(",", "."))
            val = float(r.find("Value").text.replace(",", "."))
            rows.append((date, val / nom, nom))  # RUB за 1 единицу валюты
        rows.sort()
        return rows
    raise RuntimeError(f"{code}: ЦБ не ответил за {retries} попытки") from last


def main() -> None:
    ap = argparse.ArgumentParser(description="Выгрузка курсов ЦБ РФ")
    ap.add_argument("--from", dest="d1", default="01/01/2019", help="DD/MM/YYYY")
    ap.add_argument("--to", dest="d2", default=dt.date.today().strftime("%d/%m/%Y"),
                    help="DD/MM/YYYY, по умолчанию сегодня")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("cbr_rates.json")))
    a = ap.parse_args()

    out: dict[str, list] = {}
    for code, vid in IDS.items():
        rows = fetch(code, vid, a.d1, a.d2)
        if len(rows) < MIN_ROWS:
            raise RuntimeError(f"{code}: получено {len(rows)} строк, ожидалось >= {MIN_ROWS}")
        gaps = collections.Counter((b[0] - x[0]).days for x, b in zip(rows[:-1], rows[1:]))
        out[code] = rows
        print(f"{code}: {len(rows)} строк, {rows[0][0]} -> {rows[-1][0]}, "
              f"номиналы={sorted({n for _d, _v, n in rows})}, "
              f"типичный шаг={gaps.most_common(1)[0][0]} дн, "
              f"последний курс={rows[-1][1]:.6f} RUB")

    payload = json.dumps({k: [(str(d), v) for d, v, _n in r] for k, r in out.items()})
    pathlib.Path(a.out).write_text(payload)
    print()
    print(f"записано: {a.out}")
    print(f"SHA-256:  {hashlib.sha256(payload.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
