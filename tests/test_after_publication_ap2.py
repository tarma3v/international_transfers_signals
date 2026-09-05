import datetime as dt
import numpy as np
import pandas as pd
import pytest
from ml.data import Series
from research.after_publication_ap2_features import session_state, market_features, append_features
from research.after_publication_ap2 import residual_mask, cooldown_filter


def candles():
    return pd.DataFrame({'begin':pd.to_datetime(['2024-01-09 15:20','2024-01-09 17:50','2024-01-09 18:20','2024-01-09 18:30']),
                         'end':pd.to_datetime(['2024-01-09 15:29:59','2024-01-09 17:59:59','2024-01-09 18:20:01','2024-01-09 18:30:01']),
                         'open':[10.,11.,12.,500.],'close':[10.,11.,12.,500.],
                         'high':[10.,11.,12.,500.],'low':[10.,11.,12.,500.]})


def test_nominal_completion_and_exact_cutoff_are_enforced():
    f=candles(); day=dt.date(2024,1,9)
    s=session_state(f,day)
    assert s['n']==3 and s['last']==12.
    assert session_state(f,day,dt.time(18,25))['last']==11.
    f.loc[2,'end']=pd.Timestamp('2024-01-09 18:30')
    assert session_state(f,day)['last']==11.


def test_delayed_feed_does_not_treat_bar_close_as_client_receipt():
    f=candles(); day=dt.date(2024,1,9)
    assert session_state(f,day,feed_delay_minutes=20)['last']==11.
    changed=f.copy(); changed.loc[2:,['open','close','high','low']]*=1000
    assert session_state(f,day,feed_delay_minutes=20)==session_state(changed,day,feed_delay_minutes=20)
    with pytest.raises(ValueError): session_state(f,day,feed_delay_minutes=-1)


def small_panel():
    day=dt.date(2024,1,9)
    p=pd.DataFrame([dict(currency='TJS',date=day,announced_index=2,announced_price=2.)])
    ds=np.array([dt.date(2024,1,6),dt.date(2024,1,9),dt.date(2024,1,10),dt.date(2024,1,11)],dtype=object)
    series={'CNY':Series('CNY',ds,np.array([8.,9.,10.,999.])),
            'TJS':Series('TJS',ds,np.array([1.8,1.9,2.,999.]))}
    return p,series


def test_market_basis_uses_announced_reference_and_independent_receipt():
    p,s=small_panel()
    m=market_features(p,s,{'CNYRUB_TOM':candles()})
    assert m.cny_announced_price.iloc[0]==10.
    assert m.cny_announced_effective_date.iloc[0]=='2024-01-10'
    vol=max(np.std(np.diff(np.log([8.,9.,10.])))*1e4,1.)
    assert np.isclose(m.cny_basis_last_z.iloc[0],1e4*np.log(12/10)/vol)
    # Tomorrow's not-yet-announced999 never changes today's reference.
    s['CNY'].values[-1]=.001
    pd.testing.assert_frame_equal(m,market_features(p,s,{'CNYRUB_TOM':candles()}))


def test_future_candle_corruption_leaves_earlier_features_unchanged():
    p,s=small_panel(); f=candles()
    original=market_features(p,s,{'CNYRUB_TOM':f})
    altered=f.copy(); altered.loc[3,['open','close','high','low']]*=1000
    pd.testing.assert_frame_equal(original,market_features(p,s,{'CNYRUB_TOM':altered}))
    altered.loc[2,['open','close','high','low']]*=2
    assert not original.equals(market_features(p,s,{'CNYRUB_TOM':altered}))


def test_missing_sessions_remain_finite_rows():
    p,s=small_panel(); m=market_features(p,s,{})
    assert len(m)==len(p) and m.cny_n.iloc[0]==0 and m.local_quality.iloc[0]==0
    X,n=append_features(np.array([[1.]]),['known_change_z'],m)
    assert np.isfinite(X).all() and len(n)==X.shape[1]


def test_no_announced_cny_cannot_expose_future_record_metadata():
    p,s=small_panel()
    s['CNY']=Series('CNY',np.array([dt.date(2024,1,15)],dtype=object),np.array([999.]))
    m=market_features(p,s,{'CNYRUB_TOM':candles()})
    assert m.cny_announced_effective_date.iloc[0] is None
    assert m.cny_source_received_at.iloc[0] is None
    assert m.cny_last_missing.iloc[0]==1 and m.cny_basis_last_z.iloc[0]==0


def test_residual_training_requires_mature_labels_and_past_oos_anchor():
    dates=np.array([dt.date(2022,12,1),dt.date(2023,1,3),dt.date(2023,2,2),dt.date(2023,4,3)],dtype=object)
    p=pd.DataFrame({'date':dates})
    outcomes={'mature20':np.array([dt.date(2023,1,10),dt.date(2023,2,3),dt.date(2023,4,1),dt.date(2023,5,5)],dtype=object),
              'y20':np.ones(4)}
    a=np.array([np.nan,1.,2.,3.]); origins=np.array([dt.date.max,dt.date(2023,1,1),dt.date(2023,1,1),dt.date(2023,4,1)],dtype=object)
    mask=residual_mask(p,outcomes,dt.date(2023,4,1),a,origins)
    np.testing.assert_array_equal(mask,[False,True,False,False])
    origins[1]=dt.date(2023,1,4)
    with pytest.raises(AssertionError): residual_mask(p,outcomes,dt.date(2023,4,1),a,origins)


def test_cooldown_uses_past_signals_and_crosses_week_boundary():
    days=np.array([dt.date(2024,1,5)+dt.timedelta(days=i) for i in range(8)],dtype=object)
    f=np.ones(8,dtype=bool); cur=np.repeat('TJS',8)
    actual=cooldown_filter(f,days,cur)
    np.testing.assert_array_equal(np.where(actual)[0],[0,3,6])
    f[4:]=False
    np.testing.assert_array_equal(actual[:4],cooldown_filter(f,days,cur)[:4])
