"""AP1: actual classical experiments on the reconstructed announcement clock.

Every metric is conditional on CALENDAR-ASSUMED receipt times, not certified
historical18:00 availability. Selection precedes opened-later scorecards.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_panel import build_features, build_outcomes

OUT=Path("results/research/after_publication/ap1")
DATA=Path("data/cbr_rates_2010_2026.json")
EARLY=(2017,2018,2019,2020,2022,2023)
LATER=(2024,2025,2026)
KINDS=("logit7y","hist7y","extra7y","extra3y","local_logit7y","quantile7y")
RATES=(.25,.35,.45)
SEED=20260906


def factory(kind):
    if "logit" in kind:
        return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500))
    if "extra" in kind:
        return ExtraTreesClassifier(n_estimators=200,max_depth=8,min_samples_leaf=30,
                                    max_features=.8,n_jobs=2,random_state=SEED)
    kw=dict(max_iter=160,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,
            l2_regularization=5.,random_state=SEED)
    if "quantile" in kind:
        return HistGradientBoostingRegressor(loss="quantile",quantile=.25,**kw)
    return HistGradientBoostingClassifier(**kw)


def train_mask(panel,outcomes,origin,years):
    dates=panel.date.to_numpy()
    lower=dt.date(origin.year-years,1,1)
    cutoff=origin-dt.timedelta(days=2)
    return ((dates>=lower)&(dates<cutoff)&(outcomes["mature20"]<cutoff)
            &np.isfinite(outcomes["y5"]))


def fit_scores(panel,X,outcomes,eligible,convention):
    dates=panel.date.to_numpy();cur=panel.currency.to_numpy()
    scores={k:np.full(len(panel),np.nan) for k in KINDS};logs=[]
    for year in range(2016,max(d.year for d in dates)+1):
        origin=dt.date(year,1,1)
        te=np.array([d.year==year for d in dates])
        if not te.any():continue
        for kind in KINDS:
            tr=train_mask(panel,outcomes,origin,3 if kind=="extra3y" else 7)
            regression="quantile" in kind
            if convention=="effective" and not regression:tr &= eligible
            target=outcomes["floor5"] if regression else outcomes["y5"]
            if tr.sum()<150:raise ValueError((year,kind,"insufficient history",tr.sum()))
            assert max(outcomes["mature20"][tr])<origin-dt.timedelta(days=2)
            assert max(dates[tr])<origin
            groups=CORRIDORS if kind.startswith("local") else (None,)
            for c in groups:
                train=tr if c is None else tr&(cur==c)
                test=te if c is None else te&(cur==c)
                if not test.any():continue
                if train.sum()<60 or len(np.unique(target[train]))<2:
                    # Training-only constant fallback, never a test-fitted model.
                    pred=np.repeat(float(target[train].mean()) if train.any() else float(target[tr].mean()),test.sum())
                else:
                    model=factory(kind)
                    model.fit(X[train],target[train])
                    pred=model.predict(X[test]) if regression else model.predict_proba(X[test])[:,1]
                scores[kind][test]=pred
            logs.append({"convention":convention,"candidate":kind,"year":year,
                         "n_train":int(tr.sum()),"last_train_date":str(max(dates[tr])),
                         "last_mature20":str(max(outcomes["mature20"][tr])),
                         "n_test":int(te.sum()),"embargo_calendar_days":2})
        print(f"  {convention}: annual models {year} finished",flush=True)
    return scores,logs


def rank_policy(score,dates,currencies,rate,eligible=None,window=250,warmup=40):
    fired=np.zeros(len(score),dtype=bool)
    for c in CORRIDORS:
        history=[]
        for row in np.where(currencies==c)[0]:
            value=score[row]
            if not np.isfinite(value):continue
            gate=eligible is None or bool(eligible[row])
            if len(history)>=warmup and gate:
                threshold=float(np.quantile(history[-window:],1-rate))
                # Strict tie handling prevents constant predictions from firing
                # on every day. It is fixed before viewing later performance.
                fired[row]=value>threshold
            history.append(value)
    return fired


def sign_policy(change,dates,currencies,cooldown):
    fired=np.zeros(len(change),dtype=bool)
    for c in CORRIDORS:
        last=None
        for row in np.where(currencies==c)[0]:
            if change[row]>=0 and (last is None or (dates[row]-last).days>=cooldown):
                fired[row]=True;last=dates[row]
    return fired


def adjusted(y,fired,scope,groups,weights=None):
    if weights is None:weights=np.ones(len(y))
    w=weights*scope
    m=w*fired
    yy=np.nan_to_num(y)
    n=np.bincount(groups,weights=w);success=np.bincount(groups,weights=w*yy)
    chosen=np.bincount(groups,weights=m)
    base=np.divide(success,n,out=np.zeros_like(success),where=n>0)
    expected=float(np.dot(chosen,base))
    observed=float(np.sum(m*yy))
    return observed/expected if expected>0 else np.nan


def scorecard(panel,outcomes,fired,scope,groups):
    dates=panel.date.to_numpy();cur=panel.currency.to_numpy()
    years=np.array([d.year for d in dates])
    rows=[]
    for h in HORIZONS:
        y=outcomes[f"y{h}"];valid=scope&np.isfinite(y);active=valid&fired
        n=int(active.sum());base=float(y[valid].mean()) if valid.any() else np.nan
        hit=float(y[active].mean()) if n else np.nan
        freq=[]
        for c in CORRIDORS:
            weeks=0.
            for year in sorted(set(years[valid])):
                cm=valid&(cur==c)&(years==year)
                if cm.any():weeks+=((max(dates[cm])-min(dates[cm])).days+1)/7.
            freq.append(int((active&(cur==c)).sum())/weeks if weeks else np.nan)
        rows.append({"h":h,"n_scope":int(valid.sum()),"n_signals":n,
                     "first":str(min(dates[valid])) if valid.any() else None,
                     "last":str(max(dates[valid])) if valid.any() else None,
                     "hit_rate":hit,"base_rate":base,"pooled_lift":hit/base,
                     "adjusted_lift":adjusted(y,fired,valid,groups),
                     "frequency":float(np.nanmean(freq)),"currency_rate_min":float(np.nanmin(freq)),
                     "currency_rate_max":float(np.nanmax(freq)),
                     "symmetric_bps":float(np.nanmean(outcomes[f"sym{h}"][active])) if n else np.nan,
                     "forward_bps":float(np.nanmean(outcomes[f"forward{h}"][active])) if n else np.nan})
    return rows


def summaries(frame):
    result=frame.groupby("candidate",sort=False).agg(
        min_lift=("adjusted_lift","min"),mean_lift=("adjusted_lift","mean"),
        min_rate=("currency_rate_min","min"),max_rate=("currency_rate_max","max"),
        min_forward=("forward_bps","min"),min_symmetric=("symmetric_bps","min"))
    result["cadence_penalty"]=(1-result.min_rate).clip(lower=0)+(result.max_rate-2).clip(lower=0)
    result["selection_value"]=result.min_lift-2*result.cadence_penalty
    return result


def policies(X,names,panel,model_scores,convention):
    dates=panel.date.to_numpy();cur=panel.currency.to_numpy()
    get=lambda name:X[:,names.index(name)]
    change=get("known_change");eligible=change>=0
    signals={};raw={}
    for cd in (0,3,4):signals[f"sign_cd{cd}"]=sign_policy(change,dates,cur,cd)
    for name,values in (("change",change),("change_z",get("known_change_z")),*model_scores.items()):
        values=values.copy()
        gate=eligible if convention=="effective" else None
        if gate is not None:values[~gate]=-1e9
        raw[name]=values
        for rate in RATES:
            key=f"{name}_r{round(rate*100):02d}"
            signals[key]=rank_policy(values,dates,cur,rate,gate)
    stale=np.full(len(panel),np.nan)
    values=get("known_change_z")
    for c in CORRIDORS:
        idx=np.where(cur==c)[0]
        stale[idx[20:]]=values[idx[:-20]]
    signals["stale20_z_r35"]=rank_policy(stale,dates,cur,.35)
    return signals,raw


def paired_bootstrap(panel,outcomes,signals,scope,groups,keys,control="sign_cd3",draws=1000):
    idx=np.where(scope)[0]
    dates=panel.date.to_numpy()[idx];g=groups[idx]
    _,date_id=np.unique(dates,return_inverse=True)
    nd=int(date_id.max()+1);rng=np.random.default_rng(SEED)
    allkeys=list(dict.fromkeys([control,*keys]))
    estimates={k:np.empty((draws,len(HORIZONS))) for k in allkeys}
    for b in range(draws):
        starts=rng.integers(0,nd,size=int(np.ceil(nd/20)))
        chosen=((starts[:,None]+np.arange(20))%nd).ravel()[:nd]
        w=np.bincount(chosen,minlength=nd)[date_id]
        for j,h in enumerate(HORIZONS):
            y=outcomes[f"y{h}"][idx];valid=np.isfinite(y)
            for k in allkeys:
                estimates[k][b,j]=adjusted(y,signals[k][idx],valid,g,w)
    rows=[]
    for k in allkeys:
        for j,h in enumerate(HORIZONS):
            a=estimates[k][:,j];delta=a-estimates[control][:,j]
            rows.append({"candidate":k,"h":h,"control":control,
                         "lift_lo":float(np.nanquantile(a,.025)),"lift_hi":float(np.nanquantile(a,.975)),
                         "delta_lo":float(np.nanquantile(delta,.025)),"delta_hi":float(np.nanquantile(delta,.975)),
                         "draws":draws,"block":"20 announcement dates, circular, all currencies together"})
        delta=estimates[k].mean(axis=1)-estimates[control].mean(axis=1)
        rows.append({"candidate":k,"h":"mean","control":control,
                     "delta_lo":float(np.nanquantile(delta,.025)),"delta_hi":float(np.nanquantile(delta,.975)),"draws":draws})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    series=load(DATA);panel,X,names=build_features(series)
    dates=panel.date.to_numpy();cur=panel.currency.to_numpy()
    years=np.array([d.year for d in dates])
    _,groups=np.unique([f"{c}-{d.year}" for c,d in zip(cur,dates)],return_inverse=True)
    panel.to_csv(OUT/"announcement_panel.csv",index=False)
    (OUT/"feature_names.json").write_text(json.dumps(names,indent=2))
    meta={"date":"2026-09-06","availability":"CALENDAR-ASSUMED effective_date minus1 day at18:00 Moscow",
          "baseline_support":"own-currency announcement events only, not all calendar days",
          "selection_years":EARLY,"retrospective_years":LATER,
          "data_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
          "registered_sha256":hashlib.sha256(Path("research/after_publication_ap1_registered.md").read_bytes()).hexdigest(),
          "n_rows":len(panel),"n_features":len(names),"new_bank_execution_evidence":False}
    (OUT/"metadata.json").write_text(json.dumps(meta,indent=2))
    all_logs=[]
    for convention in ("effective","publication"):
        target_dir=OUT/convention;target_dir.mkdir(exist_ok=True)
        outcomes=build_outcomes(series,panel,convention)
        eligible=X[:,names.index("known_change")]>=0
        models,logs=fit_scores(panel,X,outcomes,eligible,convention);all_logs+=logs
        signals,raw=policies(X,names,panel,models,convention)
        common=np.logical_and.reduce([np.isfinite(s) for s in models.values()])
        early=common&np.isin(years,EARLY)&(outcomes["mature20"]<dt.date(2024,1,1))
        rows=[]
        for key,fired in signals.items():
            for row in scorecard(panel,outcomes,fired,early,groups):rows.append({"candidate":key,**row})
        early_frame=pd.DataFrame(rows);early_frame.to_csv(target_dir/"early_all_horizons.csv",index=False)
        summary=summaries(early_frame)
        # Stale controls are diagnostics, never eligible winners.
        candidates=summary[~summary.index.str.startswith("stale")]
        ranked=candidates.sort_values(["selection_value","mean_lift"],ascending=False,kind="stable")
        simple=ranked[ranked.index.str.startswith(("sign_","change_"))]
        selection={"convention":convention,"selected":str(ranked.index[0]),
                   "selected_simple":str(simple.index[0]),"selection_years":EARLY,
                   "selection_value":float(ranked.iloc[0].selection_value),
                   "selected_before_later_evaluation":True,
                   "limitations":"Calendar-assumed; opened retrospective final; no executable bank quotes"}
        # This file is written before any later performance is evaluated.
        (target_dir/"selection.json").write_text(json.dumps(selection,indent=2))
        summary.to_csv(target_dir/"early_summary.csv")
        later=common&np.isin(years,LATER)
        rows=[]
        for key,fired in signals.items():
            for row in scorecard(panel,outcomes,fired,later,groups):rows.append({"candidate":key,**row})
        final=pd.DataFrame(rows);final.to_csv(target_dir/"retrospective_all_horizons.csv",index=False)
        summaries(final).to_csv(target_dir/"retrospective_summary.csv")
        focus=list(dict.fromkeys([selection["selected"],selection["selected_simple"],"sign_cd3","stale20_z_r35"]))
        detail=[]
        scopes=[("year",str(y),later&(years==y)) for y in LATER]
        scopes += [("currency",c,later&(cur==c)) for c in CORRIDORS]
        for kind,label,scope in scopes:
            for key in focus:
                for row in scorecard(panel,outcomes,signals[key],scope,groups):
                    detail.append({"candidate":key,"slice_kind":kind,"slice":label,**row})
        pd.DataFrame(detail).to_csv(target_dir/"breakdown.csv",index=False)
        for control in dict.fromkeys(["sign_cd3",selection["selected_simple"]]):
            boot=paired_bootstrap(panel,outcomes,signals,later,groups,focus,control)
            boot.to_csv(target_dir/f"paired_vs_{control}.csv",index=False)
        saved={f"score_{k}":v for k,v in raw.items()}
        saved.update({f"fired_{k}":v for k,v in signals.items()})
        saved.update({k:v for k,v in outcomes.items() if not k.startswith("mature")})
        saved.update({"common":common,"early_scope":early,"later_scope":later})
        np.savez_compressed(target_dir/"outputs.npz",**saved)
        print("SELECTION",json.dumps(selection),flush=True)
        print(final[final.candidate.isin(focus)].to_string(index=False),flush=True)
    pd.DataFrame(all_logs).to_csv(OUT/"training_log.csv",index=False)


if __name__=="__main__":main()
