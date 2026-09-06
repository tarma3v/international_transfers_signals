import numpy as np

from research.after_publication_ap29_effective import CANDIDATE, policy_inputs


def test_policy_inputs_keep_registered_expert_roles():
    old = {
        'input__primary': np.array([1., 2.]),
        'input__reserve': np.array([3., 4.]),
    }
    ap26 = {'pace_score__y20_shrink200': np.array([5., 6.])}
    ap23 = {'pace_score__soft730_pace': np.array([7., 8.])}
    inputs = policy_inputs(old, ap26, ap23)
    np.testing.assert_array_equal(inputs['primary'], [1., 2.])
    np.testing.assert_array_equal(inputs['pace'], [5., 6.])
    np.testing.assert_array_equal(inputs['backstop'], [7., 8.])
    np.testing.assert_array_equal(inputs['reserve'], [3., 4.])
    assert CANDIDATE == 's200_comp95_r60_backstop_month24_cap2'
