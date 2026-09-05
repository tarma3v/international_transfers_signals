import datetime as dt
import numpy as np
import pandas as pd

from research.round7_direct_pairs import session_state, fit_scores


def test_session_uses_nominal_completion_and_rejects_later_bars():
    # Last trade before cutoff does not mean that the ten-minute bar is final.
    f=pd.DataFrame([
        ('2025-01-10 14:50','2025-01-10 14:59:59',10.,10.,10.,10.),
        ('2025-01-10 15:00','2025-01-10 15:00:01',11.,11.,11.,11.),
        ('2025-01-10 15:10','2025-01-10 15:19:59',100.,100.,100.,100.),
    ],columns=['begin','end','open','high','low','close'])
    f['begin']=pd.to_datetime(f.begin);f['end']=pd.to_datetime(f.end)
    state=session_state(f,dt.date(2025,1,10),dt.time(15,5))
    assert state['n']==1
    assert state['last']==10.


def test_weekend_has_missing_today_and_last_prior_quote():
    f=pd.DataFrame([('2025-01-10 18:50','2025-01-10 18:59:59',10.,10.,10.,10.)],
                   columns=['begin','end','open','high','low','close'])
    f['begin']=pd.to_datetime(f.begin);f['end']=pd.to_datetime(f.end)
    state=session_state(f,dt.date(2025,1,11))
    assert state['n']==0 and np.isnan(state['mean'])
    assert state['previous']==10. and state['previous_age']>20


def test_outcome_flip_cannot_change_earlier_classifier():
    rng=np.random.default_rng(41)
    days=[dt.date(2022,1,1)+dt.timedelta(days=i) for i in range(1461)]
    dates=np.repeat(np.array(days,dtype=object),2)
    currencies=np.tile(['KZT','AMD'],len(days))
    reach=dates+dt.timedelta(days=5)
    X=rng.normal(size=(len(dates),3));y=(X[:,0]+rng.normal(size=len(dates))>0).astype(float)
    a,_=fit_scores('logit',X,y,dates,reach,currencies)
    modified=y.copy();modified[reach>=dt.date(2025,1,1)]=1-modified[reach>=dt.date(2025,1,1)]
    b,_=fit_scores('logit',X,modified,dates,reach,currencies)
    np.testing.assert_array_equal(a[dates<dt.date(2025,1,1)],b[dates<dt.date(2025,1,1)])
