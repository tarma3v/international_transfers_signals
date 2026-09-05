"""Uncertainty, cadence, paired feature ablations; no winner re-selection."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from research.after_publication_ap1 import HORIZONS, SEED, paired_bootstrap, scorecard
from research.after_publication_ap2 import OUT, KINDS


def main(output=OUT):
    OUT = output
    panel = pd.read_csv(OUT/'announcement_panel.csv')
    panel['date'] = pd.to_datetime(panel.date)
    selection = json.loads((OUT/'selection.json').read_text())
    keys = list(dict.fromkeys([selection['selected'],selection['selected_simple'],
                             'change_z_r25','cny_last_r25','stale20_cny_r35',
                             *[kind+'_r25' for kind in KINDS]]))
    with np.load(OUT/'outputs.npz') as saved:
        scope=saved['later']; idx=np.where(scope)[0]
        _,date_id=np.unique(panel.date.to_numpy()[idx],return_inverse=True)
        nd=int(date_id.max()+1); rng=np.random.default_rng(SEED)
        weights=[]
        for _ in range(1000):
            starts=rng.integers(0,nd,size=int(np.ceil(nd/20)))
            chosen=((starts[:,None]+np.arange(20))%nd).ravel()[:nd]
            weights.append(np.bincount(chosen,minlength=nd)[date_id])
        weights=np.array(weights); benefits=[]; clusters=[]
        for key in keys:
            signal=saved['signal__'+key]
            for h in HORIZONS:
                valid=np.isfinite(saved[f'y{h}'][idx])
                for metric in ('sym','forward'):
                    values=saved[f'{metric}{h}'][idx]
                    usable=signal[idx]&valid&np.isfinite(values)
                    totals=weights[:,usable].sum(axis=1)
                    means=np.divide(weights[:,usable]@values[usable],totals,
                                    out=np.full(1000,np.nan),where=totals>0)
                    benefits.append({'candidate':key,'h':h,'metric':metric,
                        'mean_bps':float(values[usable].mean()),'ci_lo':float(np.nanquantile(means,.025)),
                        'ci_hi':float(np.nanquantile(means,.975)),'n':int(usable.sum())})
            valid=scope&np.isfinite(saved['y5'])
            for c in sorted(panel.currency.unique()):
                dates=panel.loc[valid&(panel.currency==c),'date']
                selected=panel.loc[valid&signal&(panel.currency==c),'date']
                first,last=dates.min(),dates.max()
                months=pd.period_range(first.to_period('M'),last.to_period('M'),freq='M')
                counts=selected.dt.to_period('M').value_counts().reindex(months,fill_value=0)
                complete=np.array([m.start_time>=first and m.end_time.normalize()<=last for m in months])
                weeks=pd.period_range(first.to_period('W'),last.to_period('W'),freq='W')
                wc=selected.dt.to_period('W').value_counts().reindex(weeks,fill_value=0)
                gaps=selected.sort_values().diff().dt.days.dropna()
                clusters.append({'candidate':key,'currency':c,'n_signals':len(selected),
                    'empty_complete_months':int((counts.to_numpy()[complete]==0).sum()),
                    'weekly_max':int(wc.max()),'weekly_zero_share':float((wc==0).mean()),
                    'calendar_gap_max':int(gaps.max()) if len(gaps) else None})
        pd.DataFrame(benefits).to_csv(OUT/'benefit_uncertainty.csv',index=False)
        pd.DataFrame(clusters).to_csv(OUT/'clustering.csv',index=False)
        pp=panel.copy(); pp['date']=pp.date.dt.date
        signals={k.removeprefix('signal__'):saved[k] for k in saved.files if k.startswith('signal__')}
        outcomes={k:saved[k] for k in saved.files if k.startswith(('y','sym','forward','floor'))}
        ablations=[]
        for base,candidates in [('cbr_hist_r25',['market_hist_r25']),
                                ('local_ridge_r25',['residual_w25_r25','residual_w50_r25','residual_w100_r25'])]:
            b=paired_bootstrap(pp,outcomes,signals,scope,saved['groups'],candidates,base)
            ablations.append(b)
        pd.concat(ablations).to_csv(OUT/'paired_model_ablations.csv',index=False)
        detail=[]
        years=np.array([d.year for d in pp.date])
        for key in keys:
            for group,m in [(str(y),years==y) for y in (2024,2025,2026)]+[(c,pp.currency.to_numpy()==c) for c in sorted(pp.currency.unique())]:
                detail.extend({'candidate':key,'group':group,**r} for r in
                              scorecard(pp,outcomes,signals[key],scope&m,saved['groups']))
        pd.DataFrame(detail).to_csv(OUT/'diagnostic_breakdown.csv',index=False)
    market=pd.read_csv(OUT/'market_panel.csv')
    market['year']=pd.to_datetime(market.date).dt.year
    cutoff=pd.to_datetime(market.date)+pd.Timedelta(hours=18,minutes=30)
    for source in ('cny_max_source','local_max_source'):
        times=pd.to_datetime(market[source])
        assert (times[times.notna()]<=cutoff[times.notna()]).all()
    assert (pd.to_datetime(market.cny_source_received_at,utc=True)<=pd.to_datetime(market.decision_at,utc=True)).all()
    coverage=market.groupby(['currency','year']).agg(rows=('cny_n','size'),
        cny_available=('cny_last_missing',lambda x:1-x.mean()),
        local_available=('local_last_missing',lambda x:1-x.mean()),
        local_quality_mean=('local_quality','mean'),cny_age_median=('cny_age','median'))
    coverage.to_csv(OUT/'market_coverage.csv')
    print(pd.DataFrame(benefits).query('h == 5').to_string(index=False))
    print(pd.DataFrame(clusters).groupby('candidate').agg(max_gap=('calendar_gap_max','max'),
          max_week=('weekly_max','max'),max_empty_months=('empty_complete_months','max')).to_string())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--delay20',action='store_true')
    args=parser.parse_args()
    main(OUT.parent/'ap2_delay20' if args.delay20 else OUT)
