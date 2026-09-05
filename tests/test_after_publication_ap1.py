import datetime as dt

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, Series
from research.after_publication_clock import calendar_assumed_records, snapshot
from research.after_publication_panel import build_features, build_outcomes, REFERENCES
from research.after_publication_ap1 import rank_policy, train_mask, adjusted


def sample_series():
    announcements=[]
    day=dt.date(2022,1,1)
    while len(announcements)<360:
        if day.weekday()<5:announcements.append(day)
        day+=dt.timedelta(days=1)
    dates=np.array([d+dt.timedelta(days=1) for d in announcements],dtype=object)
    rng=np.random.default_rng(4066)
    return {c:Series(c,dates.copy(),np.exp(np.cumsum(rng.normal(0,.01,len(dates)))))
            for c in CORRIDORS+REFERENCES}


def test_actual_snapshot_matches_panel_and_dates_are_decision_dates():
    s=sample_series();p,x,n=build_features(s,min_history=40)
    for row in p.iloc[::81].itertuples():
        decision=dt.datetime.fromisoformat(row.decision_at)
        state=snapshot(calendar_assumed_records(s[row.currency]),decision)
        assert row.current_index==state.current_effective.source_index
        assert row.announced_index==state.latest_announced.source_index
        assert row.current_price==state.current_effective.value_rub_per_unit
        assert row.announced_price==state.next_announced.value_rub_per_unit
        assert decision.date()==row.date
        assert state.next_announced.received_at<=decision
        assert s[row.currency].dates[row.current_index]<=row.date<s[row.currency].dates[row.announced_index]
    np.testing.assert_allclose(x[:,n.index("dow_sin")],
                               [np.sin(2*np.pi*d.weekday()/7) for d in p.date])


def test_future_price_corruption_cannot_change_features_before_cutoff():
    s=sample_series();p,x,n=build_features(s,min_history=40)
    cutoff=sorted(set(p.date))[180]
    changed={}
    for c,source in s.items():
        values=source.values.copy()
        unavailable=np.array([day-dt.timedelta(days=1)>cutoff for day in source.dates])
        values[unavailable]*=100+np.arange(unavailable.sum())
        changed[c]=Series(c,source.dates.copy(),values)
    q,z,names=build_features(changed,min_history=40)
    assert n==names
    pd.testing.assert_frame_equal(p[p.date<=cutoff],q[q.date<=cutoff])
    np.testing.assert_array_equal(x[p.date<=cutoff],z[q.date<=cutoff])
    assert not np.allclose(x[p.date>cutoff],z[q.date>cutoff])


def test_two_h1_targets_are_not_silently_conflated():
    s=sample_series();p,x,n=build_features(s,min_history=40)
    effective=build_outcomes(s,p,"effective")
    publication=build_outcomes(s,p,"publication")
    np.testing.assert_array_equal(effective["y1"],x[:,n.index("known_change")]>=0)
    for i,row in enumerate(p.itertuples()):
        pos=int(row.announced_index)
        if pos+1>=len(s[row.currency]):
            assert np.isnan(publication["y1"][i]);continue
        v=s[row.currency].values
        assert publication["y1"][i]==float(v[pos+1]>=v[pos])
    valid=np.isfinite(publication["y1"])
    assert np.any(effective["y1"][valid]!=publication["y1"][valid])


def test_training_purge_resolves_full_horizon_before_embargo():
    s=sample_series();p,x,n=build_features(s,min_history=40)
    for convention in ("effective","publication"):
        outcomes=build_outcomes(s,p,convention)
        origin=dt.date(2023,1,1)
        mask=train_mask(p,outcomes,origin,7)
        assert mask.any()
        for row in p[mask].itertuples():
            source=s[row.currency]
            index=int(row.current_index if convention=="effective" else row.announced_index)
            # Independently inspect the20th future source record receipt.
            receipt=calendar_assumed_records(source)[index+20].received_at.date()
            assert receipt<origin-dt.timedelta(days=2)
            assert row.date<origin-dt.timedelta(days=2)


def test_policy_only_uses_past_scores_and_strict_ties():
    n=400
    dates=np.array([dt.date(2020,1,1)+dt.timedelta(days=i) for i in range(n)],dtype=object)
    currencies=np.repeat("TJS",n)
    values=np.random.default_rng(50).normal(size=n)
    a=rank_policy(values,dates,currencies,.35)
    values[300:]*=1000
    b=rank_policy(values,dates,currencies,.35)
    np.testing.assert_array_equal(a[:300],b[:300])
    assert not rank_policy(np.ones(n),dates,currencies,.35).any()


def test_adjusted_lift_uses_signal_matched_group_expectation():
    y=np.array([1.,1.,0.,0.,1.,0.,0.,0.])
    fired=np.array([1,1,0,0,1,0,0,0],dtype=bool)
    groups=np.array([0]*4+[1]*4)
    assert adjusted(y,fired,np.ones(8,dtype=bool),groups)==3/(2*.5+1*.25)
