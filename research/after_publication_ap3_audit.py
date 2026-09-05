"""AP3 diagnostics; later slices are descriptive, never used to reselect."""
import json
import numpy as np
import pandas as pd
from research.after_publication_ap1 import HORIZONS,SEED,scorecard
from research.after_publication_ap2_audit import main
from research.after_publication_ap3 import OUT,NORMALIZED


def paired_benefit_and_slices():
    panel=pd.read_csv(OUT/'announcement_panel.csv')
    panel.date=pd.to_datetime(panel.date).dt.date
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    control='cny_cbr_w25_r25'
    with np.load(OUT/'outputs.npz') as saved:
        idx=np.where(saved['later'])[0]
        _,day_id=np.unique(panel.date.to_numpy()[idx],return_inverse=True)
        nd=int(day_id.max()+1); rng=np.random.default_rng(SEED); weights=[]
        for _ in range(1000):
            start=rng.integers(0,nd,size=int(np.ceil(nd/20)))
            chosen=((start[:,None]+np.arange(20))%nd).ravel()[:nd]
            weights.append(np.bincount(chosen,minlength=nd)[day_id])
        weights=np.array(weights); paired=[]
        for h in HORIZONS:
            for metric in ('sym','forward'):
                values=saved[f'{metric}{h}'][idx]
                points=[]; draws=[]
                for key in (selected,control):
                    mask=saved['signal__'+key][idx]&np.isfinite(saved[f'y{h}'][idx])&np.isfinite(values)
                    n=weights[:,mask].sum(axis=1)
                    draws.append(np.divide(weights[:,mask]@values[mask],n,out=np.full(1000,np.nan),where=n>0))
                    points.append(float(values[mask].mean()))
                diff=draws[0]-draws[1]
                paired.append({'candidate':selected,'control':control,'h':h,'metric':metric,
                    'difference_bps':points[0]-points[1],
                    'ci_lo':float(np.nanquantile(diff,.025)),
                    'ci_hi':float(np.nanquantile(diff,.975)),
                    'draws':1000,'block_dates':20})
        pd.DataFrame(paired).to_csv(OUT/'paired_benefit_vs_ap2.csv',index=False)
        outcomes={k:saved[k] for k in saved.files if k.startswith(('y','sym','forward','floor'))}
        dates=pd.to_datetime(panel.date); years=dates.dt.year.to_numpy()
        slices=[]
        for year in (2024,2025,2026):
            for currency in sorted(panel.currency.unique()):
                scope=saved['later']&(years==year)&(panel.currency.to_numpy()==currency)
                slices.extend({'year':year,'currency':currency,**r} for r in
                    scorecard(panel,outcomes,saved['signal__'+selected],scope,saved['groups']))
        pd.DataFrame(slices).to_csv(OUT/'selected_year_currency.csv',index=False)
        print(pd.DataFrame(paired).to_string(index=False))


if __name__=='__main__':
    keys=[f'{name}_r25' for name in ('cny_cbr_w25','cny_last','market_hist','market_extra',
                                   'local_ridge','local_rescaled','residual_w100','residual_rescaled',*NORMALIZED)]
    keys += [f'{name}_{policy}' for name in ('cny_last','market_hist','market_extra')
             for policy in ('short_cap2','urgent_cap2')]
    keys += [f'{name}_{gate}_urgent_cap2' for name in ('cny_last','market_hist','market_extra')
             for gate in ('gate5','gate_all')]
    pairs=[('local_ridge_r25',['local_rescaled_r25','local_norm_r25']),
           ('residual_w100_r25',['residual_rescaled_r25','residual_norm100_r25']),
           ('local_norm_r25',['residual_norm25_r25','residual_norm50_r25','residual_norm100_r25']),
           ('cny_last_r25',['cny_last_short_cap2','cny_last_urgent_cap2']),
           ('market_hist_r25',['market_hist_short_cap2','market_hist_urgent_cap2'])]
    main(OUT,keys,pairs)
    paired_benefit_and_slices()
