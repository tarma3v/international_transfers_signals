"""Rebuild features on announcement events, separately from future targets.

Uses CALENDAR-ASSUMED receipts. Does not establish actual historical release
times or an executable bank price. No old next-row feature matrix is reused.
"""
from __future__ import annotations

import datetime as dt
from bisect import bisect_right

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import HORIZONS, benefit_bps, benefit_forward_only, target_now_favourable
from research.after_publication_clock import calendar_assumed_records

REFERENCES = ("USD", "CNY", "EUR")


def history_features(values):
    """Only the supplied available prefix; scales in basis points."""
    v = np.asarray(values, dtype=float)
    logv = np.log(v)
    returns = np.diff(logv)*10000
    f = {}
    for lag in (1,3,5,10,20,60):
        f[f"ret{lag}"] = float((logv[-1]-logv[max(0,len(v)-lag-1)])*10000)
    for w in (5,20,60,90,250):
        x = v[-w:]
        r = returns[-w:]
        lo,hi = float(x.min()),float(x.max())
        f[f"range{w}"] = (v[-1]-lo)/(hi-lo) if hi>lo else .5
        f[f"vol{w}"] = float(np.std(r)) if len(r) else 0.
        f[f"mean_gap{w}"] = float(10000*np.log(v[-1]/x.mean()))
        f[f"up_share{w}"] = float(np.mean(r>0)) if len(r) else .5
    return f


def build_features(series, min_history=251):
    receipts = {c: calendar_assumed_records(s) for c,s in series.items()}
    times = {c: [r.received_at for r in rows] for c,rows in receipts.items()}
    rows, features = [], []
    for c in CORRIDORS:
        own = series[c]
        for k in range(min_history,len(own)):
            event = receipts[c][k]
            decision = event.received_at
            day = decision.date()
            current = int(np.searchsorted(own.dates,day,side="right"))-1
            if current<0:
                continue
            # Prefix construction uses receipt times independently for every
            # currency. No assumption that peer/reference release dates align.
            known = {p: bisect_right(times[p],decision)-1 for p in series}
            assert known[c] == k and current<k
            values = own.values[:k+1]
            f = {"announced_"+n:v for n,v in history_features(values).items()}
            past = history_features(own.values[:current+1])
            f.update({"effective_"+n:v for n,v in past.items()})
            change = float(10000*np.log(own.values[k]/own.values[current]))
            f["known_change"] = change
            f["known_change_z"] = change/max(past["vol20"],1.)
            f["effective_age_days"] = float((day-own.dates[current]).days)
            f["next_effective_gap"] = float((event.effective_date-day).days)
            f["dow_sin"] = float(np.sin(2*np.pi*day.weekday()/7))
            f["dow_cos"] = float(np.cos(2*np.pi*day.weekday()/7))
            f["annual_sin"] = float(np.sin(2*np.pi*(day.timetuple().tm_yday-1)/365.25))
            f["annual_cos"] = float(np.cos(2*np.pi*(day.timetuple().tm_yday-1)/365.25))
            f["pre_new_year14"] = float(day.month==12 and day.day>=17)
            f["month_end"] = float(day.day>=24)
            f["after2022"] = float(day>=dt.date(2022,2,24))
            for p in CORRIDORS:
                f["currency_"+p] = float(c==p)
            peer_changes=[]
            for p in CORRIDORS+REFERENCES:
                j=known.get(p,-1)
                f[p+"_missing"] = float(j<1)
                if j<1:
                    for lag in (1,5,20):f[p+f"_ret{lag}"]=0.
                    f[p+"_source_age_days"]=365.
                    continue
                pv=series[p].values[:j+1]
                for lag in (1,5,20):
                    f[p+f"_ret{lag}"]=float(10000*np.log(pv[-1]/pv[max(0,len(pv)-lag-1)]))
                f[p+"_source_age_days"]=float((decision-receipts[p][j].received_at).total_seconds()/86400)
                if p in CORRIDORS:peer_changes.append(f[p+"_ret1"])
            f["peer_change_mean"]=float(np.mean(peer_changes)) if peer_changes else 0.
            f["peer_change_std"]=float(np.std(peer_changes)) if peer_changes else 0.
            f["local_minus_common"]=change-f["peer_change_mean"]
            latest_source=max(receipts[p][j].received_at for p,j in known.items() if j>=0)
            assert latest_source<=decision
            rows.append({"currency":c,"date":day,"decision_at":decision.isoformat(),
                         "current_index":current,"announced_index":k,
                         "effective_date":str(own.dates[current]),
                         "announced_effective_date":str(own.dates[k]),
                         "current_price":float(own.values[current]),
                         "announced_price":float(own.values[k]),
                         "source_max_received_at":latest_source.isoformat(),
                         "availability_evidence":"calendar_assumed"})
            features.append(f)
    order=sorted(range(len(rows)),key=lambda i:(rows[i]["date"],rows[i]["currency"]))
    panel=pd.DataFrame([rows[i] for i in order])
    names=sorted(features[0])
    matrix=np.array([[features[i][n] for n in names] for i in order],dtype=float)
    assert np.all(np.isfinite(matrix))
    return panel,matrix,names


def build_outcomes(series,panel,convention):
    """Outcome/receipt arrays are isolated from feature construction."""
    if convention not in ("effective","publication"):
        raise ValueError(convention)
    positions=panel.current_index if convention=="effective" else panel.announced_index
    n=len(panel)
    outcome={}
    for h in HORIZONS:
        y=np.full(n,np.nan);sym=y.copy();forward=y.copy();floor=y.copy()
        mature=np.full(n,dt.date.max,dtype=object)
        for row,(c,i) in enumerate(zip(panel.currency,positions)):
            s=series[c];i=int(i)
            if i+h>=len(s):continue
            y[row]=target_now_favourable(s.values,i,h)
            sy=benefit_bps(s.values,i,h)
            sym[row]=sy if sy is not None else np.nan
            forward[row]=benefit_forward_only(s.values,i,h)
            floor[row]=10000*np.log(s.values[i+1:i+h+1].min()/s.values[i])
            mature[row]=s.dates[i+h]-dt.timedelta(days=1)
        outcome[f"y{h}"]=y;outcome[f"sym{h}"]=sym
        outcome[f"forward{h}"]=forward;outcome[f"floor{h}"]=floor
        outcome[f"mature{h}"]=mature
    return outcome
