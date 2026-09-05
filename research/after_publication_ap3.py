"""AP3 registered normalized local/global models and sequential joint criteria."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml.data import CORRIDORS,load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA,SEED,scorecard,summaries,paired_bootstrap
from research.after_publication_panel import build_features,build_outcomes
from research.after_publication_ap2 import matured_mask,residual_mask,regressor
from research.after_publication_ap2_features import append_features
from research.after_publication_ap3_policy import (past_percentiles,sequential_policy,
    known_past_sums,future_mean_targets,utility_from_forecast)

BASE=Path('results/research/after_publication/ap2_delay20')
OUT=Path('results/research/after_publication/ap3')
NORMALIZED=('local_norm','floor_norm_hist','floor_norm_q25','residual_norm25','residual_norm50','residual_norm100')
KINDS=('r25','short_cap2','urgent_cap2')


def fit_models(panel,X,names,outcomes,scale,future):
    dates=panel.date.to_numpy(); currencies=panel.currency.to_numpy()
    scores={k:np.full(len(panel),np.nan) for k in NORMALIZED}
    origin_array=np.full(len(panel),dt.date.max,dtype=object)
    utility_pred=np.full_like(future,np.nan); logs=[]
    local_cols=[i for i,n in enumerate(names) if n.startswith(('announced_','effective_','known_',
        'market_','dow_','annual_')) or n in ('pre_new_year14','month_end','after2022')]
    floor=outcomes['floor5']/scale
    for year in range(2022,max(d.year for d in dates)+1):
        for month in (1,4,7,10):
            origin=dt.date(year,month,1)
            if origin<dt.date(2022,7,1): continue
            end=dt.date(year+1,1,1) if month==10 else dt.date(year,month+3,1)
            te=(dates>=origin)&(dates<end)
            if not te.any(): continue
            tr=matured_mask(panel,outcomes,origin)
            for c in CORRIDORS:
                train=tr&(currencies==c); test=te&(currencies==c)
                if not test.any(): continue
                if train.sum()<60:
                    pred=np.repeat(float(floor[train].mean()) if train.any() else 0.,test.sum())
                else:
                    model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
                    model.fit(X[train][:,local_cols],floor[train])
                    pred=model.predict(X[test][:,local_cols])
                scores['local_norm'][test]=pred; origin_array[test]=origin
            for name,q in (('floor_norm_hist',False),('floor_norm_q25',True)):
                if tr.sum()<400: pred=np.repeat(float(floor[tr].mean()) if tr.any() else 0.,te.sum())
                else:
                    model=regressor(q); model.fit(X[tr],floor[tr]); pred=model.predict(X[te])
                scores[name][te]=pred
            rr=residual_mask(panel,outcomes,origin,scores['local_norm'],origin_array)
            correction=np.zeros(te.sum())
            if rr.sum()>=200:
                model=regressor(); model.fit(X[rr],floor[rr]-scores['local_norm'][rr])
                correction=model.predict(X[te])
            for weight in (.25,.5,1.):
                scores[f'residual_norm{int(weight*100)}'][te]=scores['local_norm'][te]+weight*correction
            for j,h in enumerate(HORIZONS):
                if tr.sum()<400: pred=np.repeat(float(future[tr,j].mean()) if tr.any() else 0.,te.sum())
                else:
                    model=regressor(); model.set_params(loss='absolute_error')
                    model.fit(X[tr],future[tr,j]); pred=model.predict(X[te])
                utility_pred[te,j]=pred
            logs.append({'origin':str(origin),'n_train':int(tr.sum()),'last_train_mature20':str(max(outcomes['mature20'][tr])) if tr.any() else None,
                'n_test':int(te.sum()),'n_residual_train':int(rr.sum()),
                'last_residual_origin':str(max(origin_array[rr])) if rr.any() else None,
                'last_residual_mature20':str(max(outcomes['mature20'][rr])) if rr.any() else None})
            print(f'AP3 {origin}: train{tr.sum()}, genuine OOS residuals{rr.sum()}',flush=True)
    return scores,utility_pred,origin_array,logs


def build_policies(panel,saved,scores,scale,predicted_benefit):
    raw={k:saved['score__'+k] for k in ('cny_last','cny_cbr_w25','market_hist','market_extra','local_ridge','residual_w100')}
    raw['local_rescaled']=raw['local_ridge']/scale
    raw['residual_rescaled']=raw['residual_w100']/scale
    raw.update(scores)
    dates=panel.date.to_numpy(); cur=panel.currency.to_numpy()
    cny_cdf=past_percentiles(raw['cny_last'],cur)
    for name,other in [('cny_local50','local_norm'),('cny_residual50','residual_norm50'),('cny_hist50','market_hist')]:
        raw[name]=.5*cny_cdf+.5*past_percentiles(raw[other],cur)
    signals={f'{key}_{kind}':sequential_policy(values,dates,cur,kind)
             for key,values in raw.items() for kind in KINDS}
    for key in ('cny_last','market_hist','market_extra'):
        for name,gate in [('gate5',predicted_benefit[:,2]>0),('gate_all',np.min(predicted_benefit,axis=1)>0)]:
            signals[f'{key}_{name}_urgent_cap2']=sequential_policy(raw[key],dates,cur,'urgent_cap2',gate)
    assert len(signals)==57
    return raw,signals


def benefit_bootstrap(panel,outcomes,signals,scope,draws=300):
    idx=np.where(scope)[0]; _,day_id=np.unique(panel.date.to_numpy()[idx],return_inverse=True)
    nd=int(day_id.max()+1); rng=np.random.default_rng(SEED); weights=[]
    for _ in range(draws):
        start=rng.integers(0,nd,size=int(np.ceil(nd/20)))
        chosen=((start[:,None]+np.arange(20))%nd).ravel()[:nd]
        weights.append(np.bincount(chosen,minlength=nd)[day_id])
    weights=np.array(weights); rows=[]
    for key,fired in signals.items():
        for h in HORIZONS:
            y=outcomes[f'sym{h}'][idx]; usable=fired[idx]&np.isfinite(y)
            n=weights[:,usable].sum(axis=1)
            means=np.divide(weights[:,usable]@y[usable],n,out=np.full(draws,np.nan),where=n>0)
            rows.append({'candidate':key,'h':h,'mean_bps':float(y[usable].mean()) if usable.any() else np.nan,
                         'ci_lo':float(np.nanquantile(means,.025)) if usable.any() else np.nan,
                         'ci_hi':float(np.nanquantile(means,.975)) if usable.any() else np.nan})
    return pd.DataFrame(rows)


def weekly_max(panel,fired,scope):
    m=panel[scope&fired].copy()
    if m.empty: return 0
    m['week']=[(d.isocalendar()[0],d.isocalendar()[1]) for d in m.date]
    return int(m.groupby(['currency','week']).size().max())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((BASE/'metadata.json').read_text())
    for path,digest in meta['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    series=load(DATA); p,X,names=build_features(series)
    keep=p.date.to_numpy()>=dt.date(2022,1,1); p=p[keep].reset_index(drop=True); X=X[keep]
    original=pd.read_csv(BASE/'announcement_panel.csv')
    market=pd.read_csv(BASE/'market_panel.csv'); market.date=pd.to_datetime(market.date).dt.date
    assert list(zip(p.currency,p.date))==list(zip(original.currency,pd.to_datetime(original.date).dt.date))
    np.testing.assert_allclose(p.announced_price,original.announced_price,rtol=1e-12)
    p['decision_at']=market.decision_at
    X,names=append_features(X,names,market)
    assert names==json.loads((BASE/'feature_names.json').read_text())
    scale=np.maximum(X[:,names.index('announced_vol20')],1.)
    outcomes=build_outcomes(series,p,'publication')
    future=future_mean_targets(series,p,scale); past=known_past_sums(series,p)
    with np.load(BASE/'outputs.npz') as saved:
        for h in HORIZONS: np.testing.assert_array_equal(outcomes[f'y{h}'],saved[f'y{h}'])
        scores,forecast,origins,logs=fit_models(p,X,names,outcomes,scale,future)
        predicted_benefit=utility_from_forecast(forecast,scale,past)
        raw,signals=build_policies(p,saved,scores,scale,predicted_benefit)
        np.testing.assert_array_equal(signals['cny_cbr_w25_r25'],saved['signal__cny_cbr_w25_r25'])
    dates=p.date.to_numpy(); cur=p.currency.to_numpy(); years=np.array([d.year for d in dates])
    _,groups=np.unique([f'{c}-{d.year}' for c,d in zip(cur,dates)],return_inverse=True)
    # Hybrid scores need an additional past-CDF warmup; evaluation starts2023,
    # after that warmup, identically for all candidates.
    common=np.logical_and.reduce([np.isfinite(v) for v in raw.values()])&np.isfinite(forecast).all(axis=1)
    early=common&(years==2023)&(outcomes['mature20']<dt.date(2024,1,1))
    def evaluate(scope):
        return pd.DataFrame([{'candidate':k,**row} for k,s in signals.items() for row in scorecard(p,outcomes,s,scope,groups)])
    ef=evaluate(early); ef.to_csv(OUT/'early_all_horizons.csv',index=False)
    boot=benefit_bootstrap(p,outcomes,signals,early); boot.to_csv(OUT/'early_benefit_uncertainty.csv',index=False)
    summary=summaries(ef)
    summary['min_benefit_lower_ci']=boot.groupby('candidate').ci_lo.min()
    summary['max_weekly_signals']=[weekly_max(p,signals[k],early) for k in summary.index]
    summary['selection_value']=summary.min_lift-2*summary.cadence_penalty-(-summary.min_benefit_lower_ci/50).clip(lower=0)-.05*(summary.max_weekly_signals-2).clip(lower=0)
    summary['joint_early_pass']=(summary.min_lift>=1.3)&(summary.min_rate>=1)&(summary.max_rate<=2)&(summary.min_benefit_lower_ci>0)&(summary.max_weekly_signals<=2)
    ranked=summary.sort_values(['selection_value','mean_lift'],ascending=False,kind='stable')
    simple=[f'{k}_{kind}' for k in ('cny_last','cny_cbr_w25') for kind in KINDS]
    def choose(frame):
        feasible=frame[frame.joint_early_pass]
        return (feasible if len(feasible) else frame).index[0]
    selected=choose(ranked); selected_simple=choose(ranked[ranked.index.isin(simple)])
    selection={'selected':selected,'selected_simple':selected_simple,
        'selection_year':2023,'allh_mature_before':'2024-01-01',
        'joint_early_pass_count':int(summary.joint_early_pass.sum()),
        'selected_joint_early_pass':bool(summary.loc[selected,'joint_early_pass']),
        'selected_before_later_scorecard':True,'conditional_availability':'AP2-D20 unchanged; not timestamp certified'}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    summary.to_csv(OUT/'early_summary.csv')
    later=common&np.isin(years,(2024,2025,2026))
    final=evaluate(later); final.to_csv(OUT/'retrospective_all_horizons.csv',index=False)
    summaries(final).to_csv(OUT/'retrospective_summary.csv')
    focus=list(dict.fromkeys([selection['selected'],selection['selected_simple'],'cny_cbr_w25_r25','cny_last_r25','local_norm_r25','residual_norm50_r25']))
    for control in ('cny_cbr_w25_r25','cny_last_r25'):
        paired_bootstrap(p,outcomes,signals,later,groups,focus,control).to_csv(OUT/f'paired_vs_{control}.csv',index=False)
    p.to_csv(OUT/'announcement_panel.csv',index=False); market.to_csv(OUT/'market_panel.csv',index=False)
    pd.DataFrame(logs).to_csv(OUT/'training_log.csv',index=False)
    diagnostic=p[['date','currency']].copy(); diagnostic['quarter']=pd.to_datetime(diagnostic.date).dt.to_period('Q').astype(str)
    quarters=[]
    for k in ('local_ridge','local_rescaled','local_norm','residual_w100','residual_rescaled','residual_norm100','market_hist','cny_last'):
        diagnostic['score']=raw[k]; diagnostic['fired']=signals[k+'_r25']; diagnostic['scale']=scale
        f=diagnostic.groupby(['currency','quarter']).agg(median_score=('score','median'),min_score=('score','min'),max_score=('score','max'),median_scale=('scale','median'),n_signals=('fired','sum')).reset_index()
        f['candidate']=k; quarters.append(f)
    pd.concat(quarters).to_csv(OUT/'quarterly_score_diagnostics.csv',index=False)
    arrays={'dates':np.array([str(d) for d in dates]),'currencies':cur.astype(str),'groups':groups,'early':early,'later':later,
            'scale':scale,'anchor_origins':np.array([str(d) for d in origins]),'predicted_symmetric_benefit':predicted_benefit,
            'future_mean_forecast':forecast,'known_past_sums':past}
    arrays.update({f'score__{k}':v for k,v in raw.items()}); arrays.update({f'signal__{k}':v for k,v in signals.items()})
    arrays.update({k:v for k,v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT/'outputs.npz',**arrays)
    metadata={'packet':'AP3','n_rows':len(p),'n_features':len(names),'n_policies':len(signals),
        'reference':'latest announced CBR','decision':'18:30 MSK','market_delay_minutes':20,
        'calendar_assumed':True,'fresh_holdout':False,'early_bootstrap_draws':300,
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
          [DATA,BASE/'outputs.npz',BASE/'metadata.json',Path('research/after_publication_ap3_registered.md')]}}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(selection,indent=2),flush=True)
    print(final[final.candidate.isin(focus)&(final.h==5)].to_string(index=False),flush=True)


if __name__=='__main__': main()
