"""Causal score percentiles, weekly budgets and utility reconstruction for AP3."""
import numpy as np
from ml.data import CORRIDORS
from ml.targets import HORIZONS
from research.after_publication_ap1 import rank_policy


def past_percentiles(values,currencies,window=63,warmup=40):
    result=np.full(len(values),np.nan)
    for currency in CORRIDORS:
        history=[]
        for i in np.where(currencies==currency)[0]:
            value=values[i]
            if not np.isfinite(value): continue
            if len(history)>=warmup:
                reference=np.asarray(history[-window:])
                result[i]=float(np.mean(reference<value)+.5*np.mean(reference==value))
            history.append(value)
    return result


def sequential_policy(values,dates,currencies,kind,gate=None):
    if gate is None: gate=np.ones(len(values),dtype=bool)
    if kind=='r25': return rank_policy(values,dates,currencies,.25)&gate
    if kind not in ('short_cap2','urgent_cap2'): raise ValueError(kind)
    p=past_percentiles(values,currencies)
    # Strict historical quantile for the fixed-rate controller; CDF only for urgency.
    possible=rank_policy(values,dates,currencies,.35,window=63)
    result=np.zeros(len(values),dtype=bool)
    for c in CORRIDORS:
        week=None; used=0; last=None
        for i in np.where(currencies==c)[0]:
            iso=dates[i].isocalendar()[:2]
            if iso!=week: week=iso; used=0
            if used>=2 or not gate[i] or not np.isfinite(p[i]): continue
            age=(dates[i]-last).days if last is not None else 7
            if kind=='short_cap2': accept=possible[i]
            else: accept=age>=2 and p[i]>max(.45,.80-.04*age)
            if accept: result[i]=True; used+=1; last=dates[i]
    return result


def known_past_sums(series,panel):
    """Ratio sums strictly before the latest known announced price; no outcomes."""
    result=np.empty((len(panel),len(HORIZONS)))
    for row,(currency,index) in enumerate(zip(panel.currency,panel.announced_index)):
        v=series[currency].values; i=int(index)
        for j,h in enumerate(HORIZONS):
            if i<h: raise ValueError('Not enough known past prices')
            result[row,j]=v[i-h:i].sum()/v[i]
    return result


def future_mean_targets(series,panel,scale):
    """Training labels only. Never passed directly to the signal controller."""
    result=np.full((len(panel),len(HORIZONS)),np.nan)
    for row,(currency,index) in enumerate(zip(panel.currency,panel.announced_index)):
        v=series[currency].values; i=int(index)
        for j,h in enumerate(HORIZONS):
            if i+h<len(v): result[row,j]=1e4*np.log(v[i+1:i+h+1].mean()/v[i])/scale[row]
    return result


def utility_from_forecast(forecast,scale,past_sum):
    """Known past + predicted future = symmetric benefit, with numeric guard."""
    h=np.asarray(HORIZONS)
    future_ratio=np.exp(np.clip(forecast*scale[:,None]/1e4,-1,1))
    reference=(past_sum+1+h*future_ratio)/(2*h+1)
    return 1e4*(1-1/reference)
