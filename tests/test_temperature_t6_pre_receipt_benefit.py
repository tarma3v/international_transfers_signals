import numpy as np

from research.temperature_t6_pre_receipt_benefit import states_for_horizon


def test_state_mapping_uses_horizon_specific_probabilities():
    h = 5
    t4 = {'model_raw_probability_5': np.array([.1]),
          'prob__history_hist__h5': np.array([.2])}
    t5 = {}
    for cutoff in ('1030', '1130', '1230', '1330', '1430', '1500',
                   '1520', '1530'):
        t5['rank__cutoff_' + cutoff] = np.array([.3])
        t5['prob__cutoff_' + cutoff + '__h5'] = np.array([.4])
    t3 = {}
    for clock in ('1630', '1730'):
        t3['raw__update_' + clock] = np.array([.5])
        t3['prob__update_' + clock + '__h5'] = np.array([.6])
    states = states_for_horizon(h, t3, t4, t5)
    assert len(states) == 11
    assert states['premarket'][1][0] == .2
    assert states['update_1730'][1][0] == .6
