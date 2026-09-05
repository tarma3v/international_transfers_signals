"""Verify CNY units and frozen candles against scoped primary ISS requests."""
import datetime as dt
import hashlib
import json
from pathlib import Path
from research.round6_moex_data import fetch_json
from research.round6_moex_spot_1530_features import load_spot_1530_history, DATA

OUT = Path('data/after_publication_ap2')
DATES = ('2023-06-01', '2025-06-02', '2026-09-02')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    history, digest = load_spot_1530_history()
    url = 'https://iss.moex.com/iss/securities/CNYRUB_TOM.json?iss.meta=off&iss.only=description'
    meta = fetch_json(url)
    (OUT/'cny_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    description = {r[0]: r[2] for r in meta['description']['data']}
    assert description['FACEUNIT'] == 'CNY' and float(description['FACEVALUE']) == 1.
    checks = []
    for day in DATES:
        url_day = ('https://iss.moex.com/iss/engines/currency/markets/selt/boards/CETS/'
                   f'securities/CNYRUB_TOM/candles.json?from={day}&till={day}'
                   '&interval=10&iss.meta=off&iss.only=candles')
        raw = fetch_json(url_day)
        (OUT/f'cny_cets_{day}.json').write_text(json.dumps(raw, ensure_ascii=False))
        columns = raw['candles']['columns']
        fresh = {row[columns.index('begin')]: row for row in raw['candles']['data']}
        archived = [r for r in history['CNYRUB_TOM'] if str(r['begin'].date()) == day]
        mismatches = []
        for row in archived:
            key = str(row['begin'])
            if key not in fresh:
                mismatches.append(key); continue
            for field in ('open', 'close', 'high', 'low'):
                if row[field] != float(fresh[key][columns.index(field)]):
                    mismatches.append(key+':'+field)
        checks.append({'day': day, 'url': url_day, 'archived_rows': len(archived),
                       'fresh_rows': len(fresh), 'mismatches': mismatches})
        assert archived and not mismatches and len(archived) == len(fresh), checks[-1]
    manifest = {'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'archive': str(DATA), 'archive_payload_sha256': digest,
                'metadata_url': url, 'cny_quote_units': 1, 'cny_quote_currency': 'RUB',
                'spot_checks': checks, 'first_seen_certification': False,
                'usd_note_source': 'https://www.moex.com/s3933',
                'usd_note': '2026 non-deliverable ruble-settled contracts differ from historical deliverable USD; excluded from AP2 features',
                'file_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in OUT.glob('*.json') if p.name != 'manifest.json'}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
