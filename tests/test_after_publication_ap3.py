import datetime as dt
import numpy as np
import pandas as pd
from ml.data import Series
from ml.targets import HORIZONS,benefit_bps
from research.after_publication_panel import history_features
from research.after_publication_ap3_policy import (past_percentiles,sequential_policy,
    known_past_sums,future_mean_targets,utility_from_forecast)


def price_example():
    values=np.exp(np.cumsum(np.random.default_rng(54).normal(0,.003,100)))
    dates=np.array([dt.date(2022,1,1)+dt.timedelta(days=i) for i in range(100)],dtype=object)
    p=pd.DataFrame({'currency':['TJS'],'announced_index':[50]})
    return {'TJS':Series('TJS',dates,values)},p


def test_known_past_and_normalizer_do_not_read_future_prices():
    s,p=price_example()
    before=known_past_sums(s,p)
    vol=history_features(s['TJS'].values[:51])['vol20']
    target=future_mean_targets(s,p,np.array([vol]))
    s['TJS'].values[51:]*=1000
    np.testing.assert_array_equal(before,known_past_sums(s,p))
    assert vol==history_features(s['TJS'].values[:51])['vol20']
    assert not np.allclose(target,future_mean_targets(s,p,np.array([vol])))


def test_predicted_utility_identity_when_future_means_are_exact():
    s,p=price_example(); scale=np.array([max(history_features(s['TJS'].values[:51])['vol20'],1.)])
    prediction=future_mean_targets(s,p,scale)
    actual=utility_from_forecast(prediction,scale,known_past_sums(s,p))[0]
    np.testing.assert_allclose(actual,[benefit_bps(s['TJS'].values,50,h) for h in HORIZONS],atol=1e-10)


def test_true_future_utility_cannot_change_a_fixed_prediction_gate():
    s,p=price_example(); predicted=np.ones((1,5)); scale=np.array([20.])
    first=utility_from_forecast(predicted,scale,known_past_sums(s,p))>0
    s['TJS'].values[51:]*=1000
    np.testing.assert_array_equal(first,utility_from_forecast(predicted,scale,known_past_sums(s,p))>0)


def days_and_scores():
    dates=np.array([dt.date(2023,10,1)+dt.timedelta(days=i) for i in range(180)],dtype=object)
    return dates,np.repeat('TJS',len(dates)),np.arange(len(dates),dtype=float)


def test_sequential_policy_caps_each_iso_week_across_year_boundary():
    dates,cur,values=days_and_scores()
    for kind in ('short_cap2','urgent_cap2'):
        fired=sequential_policy(values,dates,cur,kind)
        counts={}
        for d in dates[fired]:
            key=d.isocalendar()[:2]; counts[key]=counts.get(key,0)+1
        assert max(counts.values())<=2
        assert any(d.year==2023 for d in dates[fired]) and any(d.year==2024 for d in dates[fired])
        if kind=='urgent_cap2': assert min((b-a).days for a,b in zip(dates[fired][:-1],dates[fired][1:]))>=2


def test_future_scores_and_gates_cannot_change_earlier_signals():
    dates,cur,values=days_and_scores(); gate=np.ones(len(dates),bool)
    for kind in ('r25','short_cap2','urgent_cap2'):
        expected=sequential_policy(values,dates,cur,kind,gate)
        changed=values.copy(); changed[100:]*=-1000
        altered=gate.copy(); altered[100:]=False
        np.testing.assert_array_equal(expected[:100],sequential_policy(changed,dates,cur,kind,altered)[:100])


def test_rejected_gate_does_not_use_the_week_budget():
    dates,cur,values=days_and_scores()
    monday=next(i for i,d in enumerate(dates) if i>50 and d.weekday()==0)
    gate=np.zeros(len(dates),bool); gate[monday+2:monday+5]=True
    for kind in ('short_cap2','urgent_cap2'):
        fired=sequential_policy(values,dates,cur,kind,gate)
        second=monday+(4 if kind=='urgent_cap2' else 3)
        assert fired[monday+2] and fired[second]
        assert fired.sum()==2
        assert not fired[:monday+2].any()


def test_constant_scores_get_midrank_and_not_all_signals():
    dates,cur,values=days_and_scores(); values[:]=1.
    p=past_percentiles(values,cur)
    np.testing.assert_array_equal(p[40:],np.repeat(.5,len(values)-40))
    assert not sequential_policy(values,dates,cur,'short_cap2').any()
    assert sequential_policy(values,dates,cur,'urgent_cap2').sum()<len(dates)//3
