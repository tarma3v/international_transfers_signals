"""Additional mandatory AP1 benefit/cluster audits; does not select models."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from research.after_publication_ap1 import OUT, HORIZONS, SEED


def main():
    panel=pd.read_csv(OUT/"announcement_panel.csv")
    panel["date"]=pd.to_datetime(panel.date)
    all_cases=[]
    for convention in ("effective","publication"):
        root=OUT/convention
        selection=json.loads((root/"selection.json").read_text())
        keys=list(dict.fromkeys([selection["selected"],selection["selected_simple"],"sign_cd3","stale20_z_r35"]))
        with np.load(root/"outputs.npz") as saved:
            scope=saved["later_scope"].astype(bool);idx=np.where(scope)[0]
            _,date_id=np.unique(panel.date.to_numpy()[idx],return_inverse=True)
            nd=int(date_id.max()+1);rng=np.random.default_rng(SEED)
            weights=[]
            for _ in range(1000):
                starts=rng.integers(0,nd,size=int(np.ceil(nd/20)))
                chosen=((starts[:,None]+np.arange(20))%nd).ravel()[:nd]
                weights.append(np.bincount(chosen,minlength=nd)[date_id])
            weights=np.array(weights)
            benefit_rows=[];cluster_rows=[]
            for key in keys:
                signal=saved["fired_"+key].astype(bool)
                for h in HORIZONS:
                    valid=np.isfinite(saved[f"y{h}"][idx])
                    active=signal[idx]&valid
                    for label in ("sym","forward"):
                        values=saved[f"{label}{h}"][idx]
                        usable=active&np.isfinite(values)
                        total=weights[:,usable].sum(axis=1)
                        means=weights[:,usable]@values[usable]/total
                        benefit_rows.append({"candidate":key,"h":h,"metric":label,
                            "mean_bps":float(values[usable].mean()),
                            "ci_lo":float(np.nanquantile(means,.025)),
                            "ci_hi":float(np.nanquantile(means,.975)),"n":int(usable.sum()),
                            "bootstrap":"1000 paired circular20-date blocks"})
                valid_all=scope&np.isfinite(saved["y5"])
                for c in sorted(panel.currency.unique()):
                    dates=panel.loc[valid_all&(panel.currency==c),"date"]
                    selected=panel.loc[valid_all&signal&(panel.currency==c),"date"]
                    first=dates.min();last=dates.max()
                    months=pd.period_range(first.to_period("M"),last.to_period("M"),freq="M")
                    counts=selected.dt.to_period("M").value_counts().reindex(months,fill_value=0)
                    complete=np.array([m.start_time>=first and m.end_time.normalize()<=last for m in months])
                    gaps=selected.sort_values().diff().dt.days.dropna()
                    weeks=pd.period_range(first.to_period("W"),last.to_period("W"),freq="W")
                    wc=selected.dt.to_period("W").value_counts().reindex(weeks,fill_value=0)
                    cluster_rows.append({"candidate":key,"currency":c,"n_signals":len(selected),
                        "complete_months":int(complete.sum()),"empty_complete_months":int((counts.to_numpy()[complete]==0).sum()),
                        "weekly_max":int(wc.max()),"weekly_zero_share":float((wc==0).mean()),
                        "calendar_gap_max":int(gaps.max()) if len(gaps) else None,
                        "calendar_gap_median":float(gaps.median()) if len(gaps) else None,
                        "share_gaps_le3":float((gaps<=3).mean()) if len(gaps) else None})
            pd.DataFrame(benefit_rows).to_csv(root/"benefit_uncertainty.csv",index=False)
            pd.DataFrame(cluster_rows).to_csv(root/"clustering.csv",index=False)
            b=pd.DataFrame(benefit_rows);cl=pd.DataFrame(cluster_rows)
            for key in keys:
                sym=b[(b.candidate==key)&(b.metric=="sym")]
                cc=cl[cl.candidate==key]
                all_cases.append({"convention":convention,"candidate":key,
                    "all_symmetric_ci_positive":bool((sym.ci_lo>0).all()),
                    "max_gap_days":int(cc.calendar_gap_max.max()),
                    "max_empty_complete_months":int(cc.empty_complete_months.max()),
                    "max_weekly_signals":int(cc.weekly_max.max())})
    pd.DataFrame(all_cases).to_csv(OUT/"benefit_clustering_summary.csv",index=False)
    print(pd.DataFrame(all_cases).to_string(index=False))


if __name__=="__main__":main()
