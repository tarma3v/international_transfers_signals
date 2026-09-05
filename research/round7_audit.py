"""Matched horizon audit and descriptive, quarterly OOF widget calibration."""
import datetime as dt
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from research.round7_direct_pairs import (
    OUT as PARENT, load_round5_features, build_targets, HORIZONS, _forward,
    _fire, POLICY, target_reach_dates, row_scores, _quarter_starts, _next_quarter,
    corridor_period_adjusted_lift, fit_scores,
)
from research.round6_multihorizon_uncertainty import _weekly_stats, _bootstrap

OUT=PARENT.parent/'audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    _X,_n,index,series,*_=load_round5_features()
    dates=np.array([r[2] for r in index],dtype=object);currencies=np.array([r[0] for r in index])
    targets=build_targets(series,index);y5=targets['fav_h5']
    forwards={h:_forward(series,index,h) for h in HORIZONS}
    selection=json.loads((PARENT/'selection.json').read_text())
    with (PARENT/'outputs.pkl').open('rb') as f:all_outputs=pickle.load(f)
    keys=list(dict.fromkeys(['incumbent',selection['selected'],selection['simple_selected'],
                            'per_currency_weights_stale20']))
    outputs={k:all_outputs[k] for k in keys}
    residual=PARENT.parent/'residual_floor'
    rs=json.loads((residual/'selection.json').read_text())['selected']
    with (residual/'outputs.pkl').open('rb') as f:ro=pickle.load(f)
    if rs!='incumbent':
        outputs['residual_'+rs]=ro[rs]
        outputs['residual_'+rs+'_stale20']=ro[rs+'_stale20']
    draws={};points={};rows=[]
    for h in HORIZONS:
        y=targets[f'fav_h{h}'];sym=targets[f'benefit_h{h}'];forward=forwards[h]
        common_valid=None
        for name,output in outputs.items():
            valid,fired=_fire(output,(2025,2026),POLICY,y,dates,currencies)
            if common_valid is None:common_valid=valid
            else:np.testing.assert_array_equal(valid,common_valid)
            lift,_,_=corridor_period_adjusted_lift(y,valid,fired,currencies,dates,(2025,2026))
            stats=_weekly_stats(y,sym,forward,valid,fired,dates,currencies)
            d=_bootstrap(stats,20260905)
            draws[name,h]=d;points[name,h]=lift
            delta=d[0]-draws['incumbent',h][0]
            rows.append({'candidate':name,'horizon':h,'lift':lift,
                         'ci_low':float(np.nanquantile(d[0],.025)),
                         'ci_high':float(np.nanquantile(d[0],.975)),
                         'delta':lift-points['incumbent',h],
                         'delta_ci_low':float(np.nanquantile(delta,.025)),
                         'delta_ci_high':float(np.nanquantile(delta,.975))})
    pd.DataFrame(rows).to_csv(OUT/'paired_horizons.csv',index=False)
    aggregates=[]
    for name in outputs:
        a=np.column_stack([draws[name,h][0] for h in HORIZONS])
        b=np.column_stack([draws['incumbent',h][0] for h in HORIZONS])
        for label,fn in (('minimum',np.min),('mean',np.mean)):
            delta=fn(a,axis=1)-fn(b,axis=1)
            aggregates.append({'candidate':name,'aggregate':label,
                'lift':float(fn([points[name,h] for h in HORIZONS])),
                'delta':float(fn([points[name,h] for h in HORIZONS])-fn([points['incumbent',h] for h in HORIZONS])),
                'delta_ci_low':float(np.nanquantile(delta,.025)),
                'delta_ci_high':float(np.nanquantile(delta,.975))})
    pd.DataFrame(aggregates).to_csv(OUT/'paired_aggregates.csv',index=False)
    # Date/quarter/availability diagnostics, without changing the selected policy.
    panel=pd.read_csv(PARENT/'direct_market_panel.csv');panel['date']=pd.to_datetime(panel.date).dt.date
    table=panel.set_index(['currency','date']).reindex([(r[0],r[2]) for r in index])
    availability=table['count'].fillna(0).to_numpy()>0
    valid_base,_=_fire(outputs['incumbent'],(2025,2026),POLICY,y5,dates,currencies)
    breakdown=[]
    for name,output in outputs.items():
        valid,fired=_fire(output,(2025,2026),POLICY,y5,dates,currencies)
        groups=[('all','all',valid),('availability','present',valid&availability),
                ('availability','absent',valid&~availability)]
        groups.extend(('currency',c,valid&(currencies==c)) for c in sorted(set(currencies)))
        for yr in (2025,2026):
            for q in range(1,5):
                groups.append(('quarter',f'{yr}Q{q}',valid&np.array([
                    d.year==yr and (d.month-1)//3+1==q for d in dates])))
        for kind,group,scope in groups:
            active=scope&fired
            if not active.any():continue
            weeks=(max(dates[scope])-min(dates[scope])).days/7
            breakdown.append({'candidate':name,'kind':kind,'group':group,
                'n_scope':int(scope.sum()),'n_signals':int(active.sum()),
                'base_rate':float(y5[scope].mean()),'hit_rate':float(y5[active].mean()),
                'pooled_lift':float(y5[active].mean()/y5[scope].mean()),
                'frequency':float(active.sum()/max(weeks,1)/(1 if kind=='currency' else 5)),
                'future_bps':float(np.nanmean(forwards[5][active]))})
    pd.DataFrame(breakdown).to_csv(OUT/'breakdown_h5.csv',index=False)
    (OUT/'period.json').write_text(json.dumps({'first':str(min(dates[valid_base])),
        'last':str(max(dates[valid_base])),'n_rows':int(valid_base.sum()),
        'bootstrap':'4000 paired moving 4-week blocks, same dates and currencies',
        'confidence_status':'conditional on fixed candidates; does not erase research selection bias'},indent=2))
    # Feasibility demonstration at the validated 15:30 snapshot only.
    raw=row_scores(outputs['incumbent'],len(index))
    cc=np.column_stack([(currencies==c).astype(float) for c in sorted(set(currencies))])
    features=np.column_stack([raw,cc]);reach=target_reach_dates(index,series,5)
    pred=np.full(len(y5),np.nan);prior=pred.copy();logs=[]
    for start in _quarter_starts():
        if start.year<2025:continue
        tr=(dates>=dt.date(2024,1,1))&(reach<start)&np.isfinite(raw)&np.isfinite(y5)
        te=(dates>=start)&(dates<_next_quarter(start))&np.isfinite(raw)&np.isfinite(y5)
        if not te.any():continue
        model=LogisticRegression(C=1.,max_iter=2000).fit(features[tr],y5[tr])
        pred[te]=model.predict_proba(features[te])[:,1];prior[te]=y5[tr].mean()
        logs.append({'quarter':str(start),'n_train':int(tr.sum()),
                     'last_resolved':str(max(reach[tr])),'intercept':float(model.intercept_[0]),
                     'coefficients':model.coef_[0].tolist()})
    scope=np.isfinite(pred)&valid_base
    f=pd.DataFrame({'date':dates[scope],'currency':currencies[scope],'rank':raw[scope],
                    'probability':pred[scope],'outcome':y5[scope],'train_prior':prior[scope]})
    f.to_csv(OUT/'widget_oof_predictions.csv',index=False)
    f['bin']=pd.cut(f.probability,np.linspace(0,1,6),include_lowest=True)
    f.groupby('bin',observed=True).agg(n=('outcome','size'),predicted=('probability','mean'),
        actual=('outcome','mean')).to_csv(OUT/'widget_reliability.csv')
    metrics={'brier_calibrated':brier_score_loss(f.outcome,f.probability),
             'brier_raw_rank':brier_score_loss(f.outcome,f['rank']),
             'brier_train_prior':brier_score_loss(f.outcome,f.train_prior),
             'n':len(f),'scope':'15:30 only, retrospective 2025-2026; no bank execution data',
             'calibration_features':['incumbent score']+sorted(set(currencies)),
             'quarterly_calibration':logs}
    (OUT/'widget_calibration.json').write_text(json.dumps(metrics,indent=2))
    print(pd.DataFrame(aggregates).to_string(index=False))
    print('WIDGET',json.dumps({k:v for k,v in metrics.items() if k!='quarterly_calibration'}))


if __name__=='__main__':main()
