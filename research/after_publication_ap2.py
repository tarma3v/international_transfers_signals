"""Frozen evening AP2 packet; models and genuinely OOS local/global residuals."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_panel import build_features, build_outcomes
from research.after_publication_ap1 import (factory, rank_policy, scorecard, summaries,
                                          paired_bootstrap, DATA, SEED)
from research.after_publication_ap2_features import load_market_frames, market_features, append_features

OUT = Path('results/research/after_publication/ap2')
KINDS = ('cbr_hist','market_hist','market_logit','market_extra','multi_hist','floor_quantile',
         'local_ridge','residual_w25','residual_w50','residual_w100')


def matured_mask(panel,outcomes,origin):
    cutoff = origin-dt.timedelta(days=2)
    return ((panel.date.to_numpy()>=dt.date(2022,1,1)) & (panel.date.to_numpy()<cutoff)
            & (outcomes['mature20']<cutoff) & np.isfinite(outcomes['y20']))


def residual_mask(panel,outcomes,origin,anchor,anchor_origins):
    eligible = matured_mask(panel,outcomes,origin) & np.isfinite(anchor)
    dates = panel.date.to_numpy()
    assert all(anchor_origins[i] <= dates[i] < origin for i in np.where(eligible)[0])
    return eligible


def regressor(quantile=False):
    kw = dict(max_iter=160,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,
              l2_regularization=5.,early_stopping=False,random_state=SEED)
    return HistGradientBoostingRegressor(loss='quantile',quantile=.25,**kw) if quantile else HistGradientBoostingRegressor(**kw)


def fit_scores(panel,X,names,n_cbr,outcomes):
    dates = panel.date.to_numpy(); cur = panel.currency.to_numpy()
    scores = {k: np.full(len(panel),np.nan) for k in KINDS}
    origins = np.full(len(panel),dt.date.max,dtype=object)
    local_cols = [i for i,n in enumerate(names) if n.startswith(('announced_','effective_','known_',
                   'market_','dow_','annual_')) or n in ('pre_new_year14','month_end','after2022')]
    logs = []
    for year in range(2022,max(d.year for d in dates)+1):
        for month in (1,4,7,10):
            origin = dt.date(year,month,1)
            if origin < dt.date(2022,7,1): continue
            end = dt.date(year+1,1,1) if month==10 else dt.date(year,month+3,1)
            te = (dates>=origin)&(dates<end)
            if not te.any(): continue
            tr = matured_mask(panel,outcomes,origin)
            for kind in KINDS[:6]:
                xx = X[:,:n_cbr] if kind=='cbr_hist' else X
                regression = kind in ('multi_hist','floor_quantile')
                target = (np.mean([outcomes[f'y{h}'] for h in HORIZONS],axis=0) if kind=='multi_hist'
                          else outcomes['floor5'] if kind=='floor_quantile' else outcomes['y5'])
                if tr.sum()<400 or np.unique(target[tr]).size<2:
                    pred = np.repeat(float(target[tr].mean()) if tr.any() else 0.,te.sum())
                else:
                    if regression: model = regressor(kind=='floor_quantile')
                    else:
                        model = factory('logit7y' if kind=='market_logit' else 'extra7y' if kind=='market_extra' else 'hist7y')
                        if kind in ('cbr_hist','market_hist'): model.set_params(early_stopping=False)
                    model.fit(xx[tr],target[tr])
                    pred = model.predict(xx[te]) if regression else model.predict_proba(xx[te])[:,1]
                scores[kind][te] = pred
            target = outcomes['floor5']
            for c in CORRIDORS:
                train = tr&(cur==c); test = te&(cur==c)
                if not test.any(): continue
                if train.sum()<60:
                    pred = np.repeat(float(target[train].mean()) if train.any() else 0.,test.sum())
                else:
                    model = make_pipeline(StandardScaler(),Ridge(alpha=100.))
                    model.fit(X[train][:,local_cols],target[train])
                    pred = model.predict(X[test][:,local_cols])
                scores['local_ridge'][test] = pred
                origins[test] = origin
            # Only earlier, truly OOS local predictions enter correction labels.
            rr = residual_mask(panel,outcomes,origin,scores['local_ridge'],origins)
            correction = np.zeros(te.sum())
            if rr.sum()>=200:
                model = regressor()
                model.fit(X[rr],target[rr]-scores['local_ridge'][rr])
                correction = model.predict(X[te])
            for weight in (.25,.5,1.):
                scores[f'residual_w{int(weight*100)}'][te] = scores['local_ridge'][te]+weight*correction
            logs.append({'origin':str(origin),'n_train':int(tr.sum()),'last_train_mature20':str(max(outcomes['mature20'][tr])) if tr.any() else None,
                         'n_test':int(te.sum()),'n_residual_train':int(rr.sum()),
                         'last_residual_origin':str(max(origins[rr])) if rr.any() else None,
                         'last_residual_mature20':str(max(outcomes['mature20'][rr])) if rr.any() else None})
            print(f'AP2 {origin}: train {tr.sum()}, OOS residual train {rr.sum()}',flush=True)
    return scores,origins,logs


def cooldown_filter(fired,dates,currencies,days=3):
    result = fired.copy()
    for c in CORRIDORS:
        last = None
        for i in np.where(currencies==c)[0]:
            if not fired[i]: continue
            if last is not None and (dates[i]-last).days < days: result[i] = False
            else: last = dates[i]
    return result


def policies(panel,X,names,market,model_scores):
    dates = panel.date.to_numpy(); cur = panel.currency.to_numpy()
    fallback = X[:,names.index('known_change_z')]
    def cny(part):
        if part=='late': return np.where(market.cny_n.to_numpy()>0,market.cny_late_z,fallback)
        return np.where(market[f'cny_{part}_missing'].to_numpy()==0,
                        market[f'cny_basis_{part}_z'],fallback)
    main = cny('last')
    raw = {'change_z':fallback,'cny_last':main,'cny_post':cny('post'),'cny_late':cny('late')}
    quality = market.local_quality.to_numpy()
    for weight in (.25,.5,1.):
        raw[f'local_mix_w{int(weight*100)}'] = main+weight*quality*(market.local_basis_last_z.to_numpy()-main)
    raw['cny_cbr_w25'] = .75*main+.25*fallback
    raw.update(model_scores)
    signals = {}
    for key,values in raw.items():
        for rate in (.25,.35):
            fired = rank_policy(values,dates,cur,rate)
            signals[f'{key}_r{int(rate*100)}'] = fired
            if rate==.35: signals[f'{key}_r35_cd3'] = cooldown_filter(fired,dates,cur)
    stale = np.full(len(panel),np.nan)
    for c in CORRIDORS:
        idx = np.where(cur==c)[0]; stale[idx[20:]] = main[idx[:-20]]
    signals['stale20_cny_r35'] = rank_policy(stale,dates,cur,.35)
    raw['stale20_cny'] = stale
    return signals,raw


def main(output=OUT,feed_delay_minutes=0):
    # Separate directories preserve the frozen zero-delay experiment.
    OUT = Path(output)
    OUT.mkdir(parents=True,exist_ok=True)
    series = load(DATA); panel,X,names = build_features(series)
    keep = panel.date.to_numpy()>=dt.date(2022,1,1)
    panel = panel[keep].reset_index(drop=True); X = X[keep]; n_cbr = len(names)
    market = market_features(panel,series,load_market_frames(),feed_delay_minutes)
    panel['decision_at'] = market.decision_at
    X,names = append_features(X,names,market)
    outcomes = build_outcomes(series,panel,'publication')
    panel.to_csv(OUT/'announcement_panel.csv',index=False)
    market.to_csv(OUT/'market_panel.csv',index=False)
    (OUT/'feature_names.json').write_text(json.dumps(names,indent=2))
    models,origins,logs = fit_scores(panel,X,names,n_cbr,outcomes)
    signals,raw = policies(panel,X,names,market,models)
    dates = panel.date.to_numpy(); cur = panel.currency.to_numpy()
    years = np.array([d.year for d in dates])
    _,groups = np.unique([f'{c}-{d.year}' for c,d in zip(cur,dates)],return_inverse=True)
    common = np.logical_and.reduce([np.isfinite(s) for s in models.values()])
    early = common&(years==2023)&(outcomes['mature20']<dt.date(2024,1,1))
    def evaluate(scope):
        return pd.DataFrame([{'candidate':key,**r} for key,s in signals.items()
                             for r in scorecard(panel,outcomes,s,scope,groups)])
    ef = evaluate(early); ef.to_csv(OUT/'early_all_horizons.csv',index=False)
    su = summaries(ef)
    su['selection_value'] -= (-su.min_symmetric/100.).clip(lower=0)
    ranked = su[~su.index.str.startswith('stale')].sort_values(['selection_value','mean_lift'],ascending=False,kind='stable')
    simple_keys = tuple(k+'_' for k in ('change_z','cny_last','cny_post','cny_late','local_mix','cny_cbr'))
    simple = ranked[ranked.index.str.startswith(simple_keys)]
    selection = {'selected':ranked.index[0],'selected_simple':simple.index[0],
                 'selection_year':2023,'mature_before':'2024-01-01','written_before_later_scorecard':True,
                 'availability':'CALENDAR-ASSUMED; decision18:30, CBR receipt18:00; no timestamp certification',
                 'market_feed_delay_minutes':feed_delay_minutes}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    su.to_csv(OUT/'early_summary.csv')
    later = common&np.isin(years,(2024,2025,2026))
    lf = evaluate(later); lf.to_csv(OUT/'retrospective_all_horizons.csv',index=False)
    summaries(lf).to_csv(OUT/'retrospective_summary.csv')
    chosen = list(dict.fromkeys([selection['selected'],selection['selected_simple'],'change_z_r25','cny_last_r25','stale20_cny_r35']))
    breakdown = []
    for key in chosen:
        for group,m in [(str(y),years==y) for y in (2024,2025,2026)]+[(c,cur==c) for c in CORRIDORS]:
            breakdown.extend({'candidate':key,'group':group,**r} for r in scorecard(panel,outcomes,signals[key],later&m,groups))
    pd.DataFrame(breakdown).to_csv(OUT/'breakdown.csv',index=False)
    for control in dict.fromkeys(['change_z_r25',selection['selected_simple']]):
        paired_bootstrap(panel,outcomes,signals,later,groups,chosen,control).to_csv(OUT/f'paired_vs_{control}.csv',index=False)
    pd.DataFrame(logs).to_csv(OUT/'training_log.csv',index=False)
    meta = {'packet':'AP2','date':'2026-09-06','target':'publication-reference',
            'n_rows':len(panel),'n_features':len(names),'n_policies':len(signals),'calendar_assumed':True,
            'decision_time':'18:30 MSK','h_units':'next observations','market_feed_delay_minutes':feed_delay_minutes,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [DATA,Path('research/after_publication_ap2_registered.md'),Path('data/after_publication_ap2/manifest.json'),Path('data/moex_direct_pairs/manifest.json')]}}
    if feed_delay_minutes:
        p=Path('research/after_publication_ap2_delay_registered.md')
        meta['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
    arrays = {'dates':np.array([str(d) for d in dates]),'currencies':cur.astype(str),'groups':groups,
              'early':early,'later':later,'anchor_origins':np.array([str(d) for d in origins])}
    arrays.update({f'score__{k}':v for k,v in raw.items()})
    arrays.update({f'signal__{k}':v for k,v in signals.items()})
    arrays.update({k:v for k,v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT/'outputs.npz',**arrays)
    print(json.dumps(selection,indent=2),flush=True)
    print(lf[(lf.candidate.isin(chosen))&(lf.h==5)].to_string(index=False),flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--delay20',action='store_true')
    args=parser.parse_args()
    main(OUT.parent/'ap2_delay20' if args.delay20 else OUT,20 if args.delay20 else 0)
