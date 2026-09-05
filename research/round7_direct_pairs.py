"""Direct FX experts and causal common/local mixtures (registered round 7)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.round5_features import load_round5_features
from research.round5_adaptation import _outputs, _quarter_starts, _next_quarter
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_decomposition import POLICY, delayed_by_currency
from research.round6_cny_error_regime import row_scores
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_fixing_proxies import proxy_scores
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.round6_resolved_models import _fire
from research.round6_multihorizon_case_audit import corridor_period_adjusted_lift
from research.round6_uzbek_central_bank_models import _forward
from research.round7_direct_pairs_data import ROOT, PAIRS

OUT = Path('results/research/round7/direct_pairs')
BASE_PATH = Path('results/research/round6/fixing_availability_router/outputs.pkl')
WEIGHTS = (0., .10, .25, .50, .75, 1.)
SEED = 20260905


def load_frames():
    result = {}
    manifest = {r['ticker']:r for r in json.loads((ROOT/'manifest.json').read_text())}
    for c in PAIRS:
        for term in ('TOM', 'TOD'):
            ticker = f'{c}RUB_{term}'
            raw = (ROOT / f'{ticker}.json').read_bytes()
            assert hashlib.sha256(raw).hexdigest() == manifest[ticker]['sha256']
            p = json.loads(raw)
            f = pd.DataFrame(p['rows'], columns=p['columns'])
            for col in ('begin', 'end'):
                f[col] = pd.to_datetime(f[col])
            for col in ('open', 'high', 'low', 'close'):
                f[col] = f[col].astype(float) / p['facevalue']
            assert (f[['open', 'high', 'low', 'close']] > 0).all().all()
            result[ticker] = f
    return result


def session_state(frame, day, cutoff=dt.time(15, 30)):
    """Only finalized ten-minute bars, including nominal end (not last trade)."""
    stop = pd.Timestamp(dt.datetime.combine(day, cutoff))
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(10)))
    nominal_end = frame['begin'] + pd.Timedelta(minutes=10)
    usable = (frame['end'] < stop) & (nominal_end <= stop)
    today = frame[usable & (frame['begin'] >= start)]
    before = frame[usable & (frame['begin'] < pd.Timestamp(day))]
    old = before.iloc[-1] if len(before) else None
    state = {'n': len(today), 'age': 720., 'mean': np.nan, 'last': np.nan,
             'ret': 0., 'range': 0., 'previous': np.nan, 'previous_age': 720.}
    if old is not None:
        state['previous'] = float(old['close'])
        state['previous_age'] = min((stop-old['end']).total_seconds()/3600,720)
    if len(today):
        state.update(mean=float(np.exp(np.log(today['close']).mean())),
                     last=float(today.iloc[-1]['close']),
                     age=(stop-today.iloc[-1]['end']).total_seconds()/60,
                     ret=float(1e4*np.log(today.iloc[-1]['close']/today.iloc[0]['open'])),
                     range=float(1e4*np.log(today['high'].max()/today['low'].min())))
    return state


def build_panel(index, series, frames, cutoff=dt.time(15,30)):
    records = []
    for c, pos, day in index:
        if day < dt.date(2022,1,1):
            continue
        states = {t: session_state(frames[f'{c}RUB_{t}'],day,cutoff) for t in ('TOM','TOD')}
        # Availability decision only; neither price direction nor future labels choose venue.
        term = 'TOM' if states['TOM']['n'] else 'TOD'
        s = states[term]
        cbr = float(series[c].values[pos])
        basis = lambda v: float(1e4*np.log(v/cbr)) if np.isfinite(v) and v>0 else np.nan
        records.append({'currency':c,'date':day,'cbr':cbr,
                        'source':term,'mean_basis':basis(s['mean']),
                        'last_basis':basis(s['last']),
                        'previous_basis':basis(s['previous']),
                        'previous_age_hours':s['previous_age'],
                        'count':s['n'],'age_minutes':s['age'],
                        'intraday_return':s['ret'],'range_bps':s['range'],
                        'quality':min(s['n']/24,1)*np.exp(-s['age']/120) if s['n'] else 0.,
                        'hard_quality':bool(s['n']>=6 and s['age']<=60),
                        'tom_count':states['TOM']['n'],'tod_count':states['TOD']['n']})
    return pd.DataFrame(records)


def quality_summary(panel):
    p=panel.copy(); p['year']=p.date.map(lambda d:d.year)
    p['available']=p['count']>0
    return p.groupby(['currency','year']).agg(
        rows=('available','size'), available=('available','sum'),
        hard_usable=('hard_quality','sum'), coverage=('available','mean'),
        hard_coverage=('hard_quality','mean'), mean_quality=('quality','mean'),
        median_bars=('count','median'), median_age_min=('age_minutes','median'),
    ).reset_index()


def feature_causality_check(index,series,frames):
    cutoff=dt.date(2025,6,30)
    subset=[row for row in index if row[2] in (cutoff,dt.date(2025,7,1))]
    original=build_panel(subset,series,frames)
    changed={}
    for ticker,f in frames.items():
        f=f.copy()
        mask=(f['end']>=pd.Timestamp('2025-06-30 15:30'))
        f.loc[mask,['open','high','low','close']]*=100
        changed[ticker]=f
    altered=build_panel(subset,series,changed)
    pd.testing.assert_frame_equal(original[original.date<=cutoff],altered[altered.date<=cutoff])
    assert not original.equals(altered), 'positive control failed'


def fit_scores(kind,matrix,y,dates,reach,currencies,local=False):
    score=np.full(len(y),np.nan); logs=[]
    for start in _quarter_starts():
        if start.year<2023: continue
        end=_next_quarter(start)
        train=(dates>=dt.date(2022,1,1)) & (reach<start) & np.isfinite(y)
        test=(dates>=start)&(dates<end)&np.isfinite(y)
        for c in (PAIRS if local else ('all',)):
            tr=train & ((currencies==c) if local else True)
            te=test & ((currencies==c) if local else True)
            if tr.sum()<(100 if local else 500) or not te.any():continue
            if kind=='logit':
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.025,max_iter=2000,random_state=SEED))
            elif kind=='hist':
                model=HistGradientBoostingClassifier(max_iter=180,learning_rate=.03,max_leaf_nodes=5,
                        min_samples_leaf=80,l2_regularization=30,early_stopping=False,random_state=SEED)
            else:
                model=ExtraTreesClassifier(n_estimators=250,max_depth=6,min_samples_leaf=35,
                        max_features=.7,n_jobs=1,random_state=SEED)
            model.fit(matrix[tr],y[tr]);score[te]=model.predict_proba(matrix[te])[:,1]
            logs.append({'kind':kind,'currency':c,'quarter':str(start),'train':int(tr.sum()),
                         'last_resolved':str(max(reach[tr])),'test':int(te.sum())})
    return score,logs


def evaluate(output,years,targets,forwards,dates,currencies,purge=None,currency=None):
    rows=[]
    for h in HORIZONS:
        y=targets[f'fav_h{h}']
        valid,fired=_fire(output,years,POLICY,y,dates,currencies)
        if purge is not None: valid &= purge
        if currency is not None: valid &= currencies==currency
        active=valid&fired
        if not active.any():
            rows.append({'horizon':h,'case_lift':np.nan,'frequency':0.,'n_signals':0,
                         'future_benefit_bps':np.nan,'symmetric_benefit_bps':np.nan})
            continue
        lift,base,macro=corridor_period_adjusted_lift(y,valid,fired,currencies,dates,years)
        weeks=(max(dates[valid])-min(dates[valid])).days/7
        rows.append({'horizon':h,'case_lift':lift,'pooled_lift':float(y[active].mean()/y[valid].mean()),
                     'base_rate':base,'hit_rate':float(y[active].mean()),'n_signals':int(active.sum()),
                     'frequency':active.sum()/weeks/(1 if currency else 5),
                     'future_benefit_bps':float(np.nanmean(forwards[h][active])),
                     'symmetric_benefit_bps':float(np.nanmean(targets[f'benefit_h{h}'][active]))})
    return pd.DataFrame(rows)


def summary(detail):
    return detail.groupby('candidate',as_index=False).agg(
        min_lift=('case_lift','min'),mean_lift=('case_lift','mean'),
        min_future_bps=('future_benefit_bps','min'),min_sym_bps=('symmetric_benefit_bps','min'),
        frequency=('frequency','mean'))


def choose(detail):
    s=summary(detail)
    pool=s[(s.min_future_bps>0)&(s.min_sym_bps>0)&s.frequency.between(.9,2)]
    if pool.empty: return 'incumbent'
    return str(pool.sort_values(['min_lift','mean_lift','candidate'],ascending=[False,False,True]).iloc[0].candidate)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    X,names,index,series,*_=load_round5_features()
    dates=np.array([r[2] for r in index],dtype=object)
    currencies=np.array([r[0] for r in index],dtype=object)
    frames=load_frames()
    panel=build_panel(index,series,frames)
    panel.to_csv(OUT/'direct_market_panel.csv',index=False)
    quality_summary(panel).to_csv(OUT/'coverage.csv',index=False)
    feature_causality_check(index,series,frames)
    table=panel.set_index(['currency','date']).reindex([(r[0],r[2]) for r in index])
    available=table['count'].fillna(0).to_numpy()>0
    hard=table['hard_quality'].eq(True).to_numpy(bool)
    quality=table['quality'].fillna(0).to_numpy(float)
    targets=build_targets(series,index); y=targets['fav_h5']
    reach=np.array(target_reach_dates(index,series,5),dtype=object)
    reach20=np.array(target_reach_dates(index,series,20),dtype=object)
    purge=reach20<dt.date(2025,1,1)
    forwards={h:_forward(series,index,h) for h in HORIZONS}
    with BASE_PATH.open('rb') as f: baseline=pickle.load(f)['availability_route']
    base=row_scores(baseline,len(index))
    direct={}
    for name in ('mean_basis','last_basis'):
        raw=table[name].to_numpy(float)
        direct[name]=causal_percentiles(raw,dates,currencies)
    candidates={'incumbent':baseline}; scores={'incumbent':base}
    specs={}
    for name,rank in direct.items():
        for gate,g in (('any',available.astype(float)),('hard',hard.astype(float)),('soft',quality)):
            for w in WEIGHTS[1:]:
                key=f'{name}_{gate}_w{int(w*100):03d}'
                v=np.where(g>0,(1-w*g)*base+w*g*np.nan_to_num(rank,nan=.5),base)
                scores[key]=v;specs[key]={'source':name,'gate':gate,'weight':w}
                candidates[key]=_outputs(v,y,dates)
    screen=[]
    for name,o in candidates.items():
        d=evaluate(o,(2024,),targets,forwards,dates,currencies,purge=purge)
        d['candidate']=name;screen.append(d)
    screen=pd.concat(screen,ignore_index=True)
    simple_choice=choose(screen)
    # Each currency selects from exactly the same frozen grid using only 2024.
    local_choices={}; local_score=base.copy(); local_screens=[]
    for c in PAIRS:
        parts=[]
        for name,o in candidates.items():
            d=evaluate(o,(2024,),targets,forwards,dates,currencies,purge=purge,currency=c)
            d['candidate']=name;parts.append(d)
        part=pd.concat(parts);local_choices[c]=choose(part)
        local_score[currencies==c]=scores[local_choices[c]][currencies==c]
        part['currency']=c;local_screens.append(part)
    candidates['per_currency_weights']=_outputs(local_score,y,dates)
    scores['per_currency_weights']=local_score
    pd.concat(local_screens).to_csv(OUT/'currency_weight_screen.csv',index=False)
    # Compact learned models, with common and local market components.
    _,_,references=load_broad_features(index,series)
    cny_history,_=load_spot_1530_history()
    common=proxy_scores(index,cny_history,references)[:,1]
    common_rank=causal_percentiles(common,dates,currencies)
    cols=['mean_basis','last_basis','previous_basis','previous_age_hours','count',
          'age_minutes','intraday_return','range_bps','quality']
    local_matrix=np.nan_to_num(table[cols].to_numpy(float),nan=0.,posinf=0.,neginf=0.)
    local_matrix[:,:3]=np.clip(local_matrix[:,:3],-5000,5000)
    static_idx=[i for i,n in enumerate(names) if n.startswith('currency_') or n in (
        'pct_range_30','pct_range_90','pct_range_180','ret_1','ret_5','ret_20')]
    static=np.column_stack([X[:,static_idx],common,common_rank])
    matrix=np.column_stack([static,local_matrix,available,hard,
                            local_matrix[:,0]-common,local_matrix[:,1]-common])
    stale_local=delayed_by_currency(local_matrix,index,rows=20)
    stale_available=delayed_by_currency(available[:,None].astype(float),index,rows=20)[:,0]
    stale_hard=delayed_by_currency(hard[:,None].astype(float),index,rows=20)[:,0]
    stale_matrix=np.column_stack([static,stale_local,stale_available,stale_hard,
                                  stale_local[:,0]-common,stale_local[:,1]-common])
    all_logs=[]
    for kind,local in (('logit',False),('hist',False),('extra',False),('logit',True)):
        key=('local_' if local else 'global_')+kind
        raw,logs=fit_scores(kind,matrix,y,dates,reach,currencies,local)
        all_logs.extend(logs)
        rank=causal_percentiles(raw,dates,currencies)
        for w in (.25,.50,1.):
            name=f'{key}_w{int(w*100)}'
            score=np.where(available,(1-w)*base+w*np.nan_to_num(rank,nan=.5),base)
            scores[name]=score;candidates[name]=_outputs(score,y,dates)
        stale,_=fit_scores(kind,stale_matrix,y,dates,reach,currencies,local)
        np.savez_compressed(OUT/f'{key}_scores.npz',raw=raw,stale=stale)
        print('fitted',key,flush=True)
    pd.DataFrame(all_logs).to_csv(OUT/'training_log.csv',index=False)
    # Register a deliberately stale local control with unchanged live routing.
    for name in ('per_currency_weights',simple_choice):
        stale_score=base.copy()
        for c in PAIRS:
            spec=specs.get(local_choices[c] if name=='per_currency_weights' else name)
            if spec is None: continue
            raw=table[spec['source']].to_numpy(float)
            sr=causal_percentiles(delayed_by_currency(raw[:,None],index,rows=20)[:,0],dates,currencies)
            g={'any':available.astype(float),'hard':hard.astype(float),'soft':quality}[spec['gate']]
            w=spec['weight'];mask=currencies==c
            mixed=np.where(g>0,(1-w*g)*base+w*g*np.nan_to_num(sr,nan=.5),base)
            stale_score[mask]=mixed[mask]
        candidates[name+'_stale20']=_outputs(stale_score,y,dates)
    screen=[]
    for name,o in candidates.items():
        if name.endswith('_stale20'): continue
        d=evaluate(o,(2024,),targets,forwards,dates,currencies,purge=purge)
        d['candidate']=name;screen.append(d)
    screen=pd.concat(screen,ignore_index=True)
    selected=choose(screen)
    selection={'selected':selected,'simple_selected':simple_choice,
               'per_currency':local_choices,'specs':specs,'screen_purged_h20':True,
               'selection_period':'2024 (h20 resolved before 2025)',
               'later_status':'previously explored retrospective 2025-2026',
               'feature_causality_check':True,'policy':POLICY}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    screen.to_csv(OUT/'screen_2024_by_horizon.csv',index=False)
    summary(screen).to_csv(OUT/'screen_2024_summary.csv',index=False)
    print('FROZEN',json.dumps({k:v for k,v in selection.items() if k!='specs'}),flush=True)
    # Later results are opened only after selection was persisted.
    later=[]; breakdown=[]
    for name,o in candidates.items():
        for period,years in (('2025',(2025,)),('2026',(2026,)),('2025-2026',(2025,2026))):
            d=evaluate(o,years,targets,forwards,dates,currencies)
            d['candidate']=name;d['period']=period;later.append(d)
        if name in ('incumbent',simple_choice,selected,'per_currency_weights'):
            for c in PAIRS:
                d=evaluate(o,(2025,2026),targets,forwards,dates,currencies,currency=c)
                d['candidate']=name;d['currency']=c;breakdown.append(d)
    later=pd.concat(later,ignore_index=True)
    later.to_csv(OUT/'later_by_horizon.csv',index=False)
    pd.concat(breakdown).to_csv(OUT/'currency_breakdown.csv',index=False)
    with (OUT/'outputs.pkl').open('wb') as f:pickle.dump(candidates,f)
    print(later[(later.period=='2025-2026')&(later.horizon==5)][[
        'candidate','case_lift','frequency','future_benefit_bps']].to_string(index=False))


if __name__=='__main__': main()
