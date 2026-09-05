"""Evening completed-bar features relative to the already ANNOUNCED CBR rate."""
from __future__ import annotations
import datetime as dt
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from research.after_publication_clock import calendar_assumed_records
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.round7_direct_pairs import load_frames

CUTOFF = dt.time(18, 30)
MOSCOW = ZoneInfo('Europe/Moscow')


def load_market_frames():
    root = Path('data/after_publication_ap2')
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['file_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    history, digest = load_spot_1530_history()
    assert digest == manifest['archive_payload_sha256']
    meta = json.loads((root/'cny_metadata.json').read_text())
    desc = {r[0]: r[2] for r in meta['description']['data']}
    face = float(desc['FACEVALUE'])
    frames = load_frames()
    frame = pd.DataFrame(history['CNYRUB_TOM'])
    for col in ('open', 'close', 'high', 'low'): frame[col] /= face
    frames['CNYRUB_TOM'] = frame
    return frames


def session_state(frame, day, cutoff=CUTOFF, feed_delay_minutes=0):
    stop = pd.Timestamp(dt.datetime.combine(day, cutoff))
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(10)))
    # end is the last actual trade, not necessarily bar finalization.
    if feed_delay_minutes < 0: raise ValueError('Feed delay cannot be negative')
    delay = pd.Timedelta(minutes=feed_delay_minutes)
    available = ((frame['begin'] >= start) & (frame['end'] + delay < stop)
                 & (frame['begin'] + pd.Timedelta(minutes=10) + delay <= stop))
    f = frame[available].sort_values('begin')
    state = dict(n=0., last=np.nan, mean=np.nan, post=np.nan, late=0.,
                 age=720., quality=0., range=0., max_source=None)
    if f.empty: return state
    boundary = pd.Timestamp(dt.datetime.combine(day, dt.time(15, 30)))
    early = f[(f['end'] < boundary) & (f['begin'] + pd.Timedelta(minutes=10) <= boundary)]
    post = f[f['begin'] >= boundary]
    age = (stop-f.iloc[-1]['end']).total_seconds()/60
    last = float(f.iloc[-1]['close'])
    state.update(n=float(len(f)), last=last, mean=float(np.exp(np.log(f['close']).mean())),
                 post=float(np.exp(np.log(post['close']).mean())) if len(post) else np.nan,
                 late=float(1e4*np.log(last/early.iloc[-1]['close'])) if len(early) and len(post) else 0.,
                 age=float(age), quality=float(min(len(f)/24,1)*np.exp(-age/120)),
                 range=float(1e4*np.log(f['high'].max()/f['low'].min())),
                 max_source=(f['begin']+pd.Timedelta(minutes=10)+delay).max().isoformat())
    return state


def market_features(panel, series, frames, feed_delay_minutes=0):
    day_frames = {ticker: {day: f for day,f in frame.groupby(frame['begin'].dt.date)}
                  for ticker,frame in frames.items()}
    empty = pd.DataFrame({k: pd.Series(dtype='datetime64[ns]' if k in ('begin','end') else float)
                          for k in ('begin','end','open','close','high','low')})
    cache = {}
    def state(ticker, day):
        key = ticker,day
        if key not in cache:
            cache[key] = session_state(day_frames.get(ticker,{}).get(day,empty),day,
                                       feed_delay_minutes=feed_delay_minutes)
        return cache[key]
    receipts = calendar_assumed_records(series['CNY'])
    times = [r.received_at for r in receipts]
    records = []
    for row in panel.itertuples():
        day = row.date
        decision = dt.datetime.combine(day,CUTOFF,tzinfo=MOSCOW)
        j = bisect_right(times,decision)-1
        ref = float(series['CNY'].values[j]) if j >= 0 else np.nan
        cv = np.diff(np.log(series['CNY'].values[max(0,j-20):j+1]))*1e4
        cvol = max(float(np.std(cv)),1.) if len(cv) else 1.
        values = series[row.currency].values[:int(row.announced_index)+1]
        lvol = max(float(np.std(np.diff(np.log(values[-21:]))*1e4)),1.)
        cs = state('CNYRUB_TOM',day)
        ts = state(row.currency+'RUB_TOM',day)
        term = 'TOM' if ts['n'] else 'TOD'
        ls = ts if term == 'TOM' else state(row.currency+'RUB_TOD',day)
        rec = dict(currency=row.currency,date=day,decision_at=decision.isoformat(),
                   cny_announced_price=ref,
                   cny_announced_effective_date=str(series['CNY'].dates[j]) if j>=0 else None,
                   cny_source_received_at=receipts[j].received_at.isoformat() if j>=0 else None,
                   local_source_term=term)
        for prefix,s,reference,vol in (('cny',cs,ref,cvol),('local',ls,row.announced_price,lvol)):
            for part in ('last','mean','post'):
                value = s[part]
                valid = np.isfinite(value) and np.isfinite(reference) and reference>0
                rec[f'{prefix}_basis_{part}_z'] = float(1e4*np.log(value/reference)/vol) if valid else 0.
                rec[f'{prefix}_{part}_missing'] = float(not valid)
            for field in ('n','age','quality','range'): rec[f'{prefix}_{field}'] = s[field]
            rec[f'{prefix}_late_z'] = s['late']/vol
            rec[f'{prefix}_max_source'] = s['max_source']
        records.append(rec)
    return pd.DataFrame(records)


def append_features(X,names,market):
    columns = [c for c in market if c.startswith(('cny_','local_'))
               and pd.api.types.is_numeric_dtype(market[c]) and c != 'cny_announced_price']
    matrix = np.column_stack([X,market[columns].to_numpy(float)])
    assert np.isfinite(matrix).all()
    return matrix,[*names,*['market_'+c for c in columns]]
