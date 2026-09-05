"""Reproducible CETS direct-pair candle archive; no outcome data in collection."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

from research.round6_moex_data import fetch_json

ROOT = Path('data/moex_direct_pairs')
PAIRS = ('KZT', 'AMD', 'KGS', 'TJS', 'UZS')
FROM, TILL = '2022-01-01', '2026-09-03'


def collect(ticker):
    path = ROOT / f'{ticker}.json'
    if path.exists():
        return json.loads(path.read_text())
    meta_url = f'https://iss.moex.com/iss/securities/{ticker}.json?iss.meta=off'
    meta = fetch_json(meta_url)
    desc = {row[0]: row[2] for row in meta['description']['data']}
    face = float(desc['FACEVALUE'])
    rows, urls, start, columns = [], [], 0, None
    while True:
        query = urlencode({'from': FROM, 'till': TILL, 'interval': 10,
                           'start': start, 'iss.meta': 'off', 'iss.only': 'candles'})
        url = (f'https://iss.moex.com/iss/engines/currency/markets/selt/'
               f'boards/CETS/securities/{ticker}/candles.json?{query}')
        block = fetch_json(url)['candles']
        urls.append(url)
        if columns is not None and columns != block['columns']:
            raise ValueError('candle schema changed')
        columns = block['columns']
        page = block['data']
        rows.extend(page)
        if len(page) < 500:
            break
        start += len(page)
    begin = [row[columns.index('begin')] for row in rows]
    assert begin == sorted(set(begin)), ticker
    payload = {'ticker': ticker, 'board': 'CETS', 'currency': desc['FACEUNIT'],
               'facevalue': face, 'from': FROM, 'till': TILL,
               'retrieved_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'timezone': 'Europe/Moscow', 'interval_minutes': 10,
               'metadata_url': meta_url, 'metadata': meta,
               'columns': columns, 'rows': rows, 'urls': urls}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print(ticker, len(rows), 'candles', 'quote units', face, flush=True)
    return payload


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    tickers = [f'{c}RUB_{term}' for c in PAIRS for term in ('TOM', 'TOD')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        payloads = list(pool.map(collect, tickers))
    manifest = []
    for p in payloads:
        raw = (ROOT / f'{p["ticker"]}.json').read_bytes()
        manifest.append({'ticker': p['ticker'], 'facevalue': p['facevalue'],
                         'rows': len(p['rows']), 'sha256': hashlib.sha256(raw).hexdigest(),
                         'first': p['rows'][0][-2] if p['rows'] else None,
                         'last': p['rows'][-1][-1] if p['rows'] else None})
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
