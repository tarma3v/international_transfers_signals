"""Completed price paths only: the frozen FX volume/value columns are all null."""
import datetime as dt
import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_publication_clock import calendar_assumed_records
from research.after_publication_ap2_features import CUTOFF, MOSCOW

FIELDS = ('median3_basis_z', 'ew30_basis_z', 'return_z', 'slope_z', 'efficiency',
          'rv_z', 'positive_share', 'zero_share', 'last_jump_z', 'position',
          'drawdown_z', 'rebound_z', 'early_return_z', 'middle_return_z',
          'late_return_z', 'late_variation_share', 'grid_density', 'path_missing')


def source_missingness():
    source = Path('data/moex_spot_fx_10min_2022_2026.json')
    payload = json.loads(source.read_text())
    instruments = [i for i in payload['instruments'] if i['ticker'] == 'CNYRUB_TOM']
    items = [(source, i['ticker'], i['columns'], i['rows']) for i in instruments]
    for path in sorted(Path('data/moex_direct_pairs').glob('*RUB_*.json')):
        item = json.loads(path.read_text())
        items.append((path, item['ticker'], item['columns'], item['rows']))
    for path in sorted(Path('data/after_publication_ap2').glob('cny_cets_*.json')):
        item = json.loads(path.read_text())['candles']
        items.append((path, path.stem, item['columns'], item['data']))
    rows = []
    for path, ticker, columns, values in items:
        row = dict(source=str(path), ticker=ticker, rows=len(values),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for field in ('volume', 'value'):
            raw = [v[columns.index(field)] for v in values]
            row[field + '_non_null'] = sum(v is not None for v in raw)
        rows.append(row)
    return pd.DataFrame(rows)


def completed(frame, day):
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(10)))
    stop = pd.Timestamp(dt.datetime.combine(day, CUTOFF))
    delay = pd.Timedelta(minutes=20)
    return frame[(frame.begin >= start) & (frame.end + delay < stop)
                 & (frame.begin + pd.Timedelta(minutes=30) <= stop)].sort_values('begin')


def path_shape(frame, day, reference, scale):
    if not np.isfinite(scale) or scale < 1 or not np.isfinite(reference) or reference <= 0:
        raise ValueError('Known positive reference and causal scale >=1 required')
    f = completed(frame, day)
    result = dict.fromkeys(FIELDS, 0.)
    result.update(position=.5, path_missing=1.)
    info = dict(n=0, first_begin=None, last_end=None, max_available=None,
                median3_price=None, ew30_price=None, reference=float(reference), scale=float(scale))
    if f.empty:
        return result, info
    prices = f[['open', 'close', 'high', 'low']].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Observed OHLC must be finite and positive')
    close = np.log(f.close.to_numpy(float))
    times = (f.end - f.end.iloc[0]).dt.total_seconds().to_numpy() / 60
    returns = np.diff(close) * 1e4
    elapsed = times[-1] - times
    weights = np.exp2(-elapsed / 30.)
    median = float(np.median(close[-3:]))
    ew = float(np.dot(weights, close) / weights.sum())
    span = times[-1]
    slope = np.dot(times - times.mean(), close - close.mean()) / np.sum((times - times.mean()) ** 2) if span > 0 else 0.
    absolute = np.abs(returns).sum()
    hi, lo = np.log(f.high.max()), np.log(f.low.min())
    late = f.begin.to_numpy()[1:] >= np.datetime64(dt.datetime.combine(day, dt.time(15, 30)))
    variation = np.square(returns).sum()
    result.update(median3_basis_z=(median - np.log(reference)) * 1e4 / scale,
        ew30_basis_z=(ew - np.log(reference)) * 1e4 / scale,
        return_z=(close[-1] - np.log(f.open.iloc[0])) * 1e4 / scale,
        slope_z=float(slope * span * 1e4 / scale),
        efficiency=float((close[-1] - close[0]) * 1e4 / absolute) if absolute > 0 else 0.,
        rv_z=float(np.sqrt(variation) / scale),
        positive_share=float((returns > 0).mean()) if len(returns) else 0.,
        zero_share=float((returns == 0).mean()) if len(returns) else 0.,
        last_jump_z=float(returns[-1] / scale) if len(returns) else 0.,
        position=float((close[-1] - lo) / (hi - lo)) if hi > lo else .5,
        drawdown_z=float((close[-1] - hi) * 1e4 / scale),
        rebound_z=float((close[-1] - lo) * 1e4 / scale),
        late_variation_share=float(np.square(returns[late]).sum() / variation) if variation > 0 else 0.,
        grid_density=float(len(f) / 49), path_missing=0.)
    for key, first, last in (('early', dt.time(10), dt.time(12)),
                             ('middle', dt.time(12), dt.time(15, 30)),
                             ('late', dt.time(15, 30), CUTOFF)):
        start = pd.Timestamp(dt.datetime.combine(day, first))
        end = pd.Timestamp(dt.datetime.combine(day, last))
        segment = f[(f.begin >= start) & (f.end < end)
                    & (f.begin + pd.Timedelta(minutes=10) <= end)]
        result[key + '_return_z'] = float(np.log(segment.close.iloc[-1] / segment.open.iloc[0]) * 1e4 / scale) if len(segment) else 0.
    available = pd.concat([f.end + pd.Timedelta(minutes=20), f.begin + pd.Timedelta(minutes=30)], axis=1).max(axis=1)
    info.update(n=len(f), first_begin=str(f.begin.iloc[0]), last_end=str(f.end.iloc[-1]),
                max_available=str(available.max()), median3_price=float(np.exp(median)), ew30_price=float(np.exp(ew)))
    return result, info


def build_shapes(panel, series, frames):
    day_frames = {ticker: {day: f for day, f in frame.groupby(frame.begin.dt.date)} for ticker, frame in frames.items()}
    empty = next(iter(frames.values())).iloc[:0]
    records, provenance = [], []
    receipts = calendar_assumed_records(series['CNY'])
    times = [r.received_at for r in receipts]
    cache = {}
    for row in panel.itertuples():
        when = dt.datetime.combine(row.date, CUTOFF, tzinfo=MOSCOW)
        ci = bisect_right(times, when) - 1
        assert ci >= 20
        values = series['CNY'].values[max(0, ci - 20):ci + 1]
        cscale = max(float(np.std(np.diff(np.log(values)) * 1e4)), 1.)
        local_values = series[row.currency].values[max(0, int(row.announced_index) - 20):int(row.announced_index) + 1]
        lscale = max(float(np.std(np.diff(np.log(local_values)) * 1e4)), 1.)
        tom = day_frames.get(row.currency + 'RUB_TOM', {}).get(row.date, empty)
        local_ticker = row.currency + ('RUB_TOM' if len(completed(tom, row.date)) else 'RUB_TOD')
        record = {'date': str(row.date), 'currency': row.currency}
        for prefix, ticker, ref, scale in (('cny', 'CNYRUB_TOM', series['CNY'].values[ci], cscale),
                                          ('local', local_ticker, row.announced_price, lscale)):
            key = (ticker, row.date, float(ref), scale)
            if key not in cache:
                cache[key] = path_shape(day_frames.get(ticker, {}).get(row.date, empty), row.date, ref, scale)
            shape, info = cache[key]
            record.update({prefix + '_' + k: v for k, v in shape.items()})
            provenance.append({'date': str(row.date), 'currency': row.currency, 'prefix': prefix, 'ticker': ticker, **info})
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(provenance)
