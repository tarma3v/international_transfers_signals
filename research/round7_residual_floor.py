"""Second predeclared family: predict future floor as a correction to CNY basis.

Uses existing market panel. Choices still use purged 2024 only; the later period
has already been opened in round 7A and is explicitly retrospective.
"""
import datetime as dt
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.round7_direct_pairs import (
    OUT as PARENT, SEED, PAIRS, _outputs, _quarter_starts, _next_quarter,
    load_round5_features, load_broad_features, load_spot_1530_history, proxy_scores,
    build_targets, HORIZONS, target_reach_dates, row_scores, causal_percentiles,
    _forward, evaluate, summary, choose, delayed_by_currency,
)

OUT=PARENT.parent/'residual_floor'


def predict(kind,matrix,floor,common,dates,reach,currencies,local=False):
    prediction=np.full(len(floor),np.nan);log=[]
    for start in _quarter_starts():
        if start.year<2023:continue
        train=(dates>=dt.date(2022,1,1))&(reach<start)&np.isfinite(floor)
        test=(dates>=start)&(dates<_next_quarter(start))&np.isfinite(floor)
        for c in (PAIRS if local else ('all',)):
            tr=train&((currencies==c) if local else True)
            te=test&((currencies==c) if local else True)
            if tr.sum()<(100 if local else 500) or not te.any():continue
            if kind=='ridge':
                model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
            elif kind=='extra':
                model=ExtraTreesRegressor(n_estimators=250,max_depth=6,min_samples_leaf=40,
                                          max_features=.7,n_jobs=1,random_state=SEED)
            else:
                extra={'loss':'quantile','quantile':.25} if kind=='quantile' else {}
                model=HistGradientBoostingRegressor(max_iter=180,learning_rate=.03,
                      max_leaf_nodes=5,min_samples_leaf=80,l2_regularization=30,
                      early_stopping=False,random_state=SEED,**extra)
            model.fit(matrix[tr],floor[tr]-common[tr])
            prediction[te]=common[te]+model.predict(matrix[te])
            log.append({'model':kind,'currency':c,'quarter':str(start),
                        'train':int(tr.sum()),'last_resolved':str(max(reach[tr]))})
    return prediction,log


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    X,names,index,series,*_=load_round5_features()
    dates=np.array([r[2] for r in index],dtype=object);currencies=np.array([r[0] for r in index])
    table=pd.read_csv(PARENT/'direct_market_panel.csv')
    table['date']=pd.to_datetime(table.date).dt.date
    table=table.set_index(['currency','date']).reindex([(r[0],r[2]) for r in index])
    _,_,refs=load_broad_features(index,series);hist,_=load_spot_1530_history()
    common=proxy_scores(index,hist,refs)[:,1]
    cols=['mean_basis','last_basis','previous_basis','previous_age_hours','count',
          'age_minutes','intraday_return','range_bps','quality']
    local=np.nan_to_num(table[cols].to_numpy(float),nan=0.)
    local[:,:3]=np.clip(local[:,:3],-5000,5000)
    stale=delayed_by_currency(local,index,rows=20)
    allowed=table['count'].fillna(0).to_numpy()>0
    selected_cols=[i for i,n in enumerate(names) if n.startswith('currency_') or n in (
        'pct_range_30','pct_range_90','pct_range_180','ret_1','ret_5','ret_20')]
    static=np.column_stack([X[:,selected_cols],common])
    matrix=np.column_stack([static,local,local[:,0]-common])
    stale_matrix=np.column_stack([static,stale,stale[:,0]-common])
    targets=build_targets(series,index);y=targets['fav_h5']
    reach=target_reach_dates(index,series,5)
    purge=target_reach_dates(index,series,20)<dt.date(2025,1,1)
    forwards={h:_forward(series,index,h) for h in HORIZONS}
    floor=np.full(len(index),np.nan)
    for row,(c,pos,day) in enumerate(index):
        v=series[c].values
        if pos+5<len(v):floor[row]=1e4*np.log(np.min(v[pos+1:pos+6])/v[pos])
    with (PARENT/'outputs.pkl').open('rb') as f:prior=pickle.load(f)
    candidates={'incumbent':prior['incumbent']};base=row_scores(candidates['incumbent'],len(index))
    controls={};logs=[]
    for kind,is_local in (('ridge',False),('hist',False),('quantile',False),('extra',False),('ridge',True)):
        key=('local_' if is_local else 'global_')+kind
        fresh,log=predict(kind,matrix,floor,common,dates,reach,currencies,is_local)
        delayed,_=predict(kind,stale_matrix,floor,common,dates,reach,currencies,is_local)
        logs.extend(log)
        rank=causal_percentiles(fresh,dates,currencies)
        stale_rank=causal_percentiles(delayed,dates,currencies)
        for w in (.10,.25,.50,1.):
            name=f'{key}_w{int(w*100)}'
            candidates[name]=_outputs(np.where(allowed,(1-w)*base+w*rank,base),y,dates)
            controls[name]=_outputs(np.where(allowed,(1-w)*base+w*stale_rank,base),y,dates)
        np.savez_compressed(OUT/f'{key}.npz',prediction=fresh,stale=delayed)
        print('residual fitted',key,flush=True)
    # Physical source check is in 7A; this independently checks mature outcome gating.
    altered=floor.copy();altered[reach>=dt.date(2025,7,1)]+=10000.
    a,_=predict('ridge',matrix,floor,common,dates,reach,currencies)
    b,_=predict('ridge',matrix,altered,common,dates,reach,currencies)
    np.testing.assert_array_equal(a[dates<dt.date(2025,7,1)],b[dates<dt.date(2025,7,1)])
    assert not np.allclose(a[dates>=dt.date(2025,7,1)],b[dates>=dt.date(2025,7,1)],equal_nan=True)
    screens=[]
    for name,o in candidates.items():
        d=evaluate(o,(2024,),targets,forwards,dates,currencies,purge=purge)
        d['candidate']=name;screens.append(d)
    screen=pd.concat(screens);selected=choose(screen)
    (OUT/'selection.json').write_text(json.dumps({'selected':selected,
        'selection_period':'purged 2024','target':'log minimum of next 5 CBR observations / current',
        'anchor':'CNY fixing basis','outcome_corruption_check':True,
        'later_status':'retrospective, opened in previous packet'},indent=2))
    print('RESIDUAL FROZEN',selected,flush=True)
    screen.to_csv(OUT/'screen_2024_by_horizon.csv',index=False)
    summary(screen).to_csv(OUT/'screen_2024_summary.csv',index=False)
    pd.DataFrame(logs).to_csv(OUT/'training_log.csv',index=False)
    if selected in controls:candidates[selected+'_stale20']=controls[selected]
    later=[]
    for name,o in candidates.items():
        for period,years in (('2025',(2025,)),('2026',(2026,)),('2025-2026',(2025,2026))):
            d=evaluate(o,years,targets,forwards,dates,currencies)
            d['candidate']=name;d['period']=period;later.append(d)
    pd.concat(later).to_csv(OUT/'later_by_horizon.csv',index=False)
    with (OUT/'outputs.pkl').open('wb') as f:pickle.dump(candidates,f)


if __name__=='__main__':main()
