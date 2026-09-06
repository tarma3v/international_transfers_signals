import numpy as np


def test_dual_gate_combination_only_rejects_double_rejection():
    precision = np.array([True, False, False])
    meta = np.array([.1, .7, .2])
    combined = np.where(precision, 1., meta)
    np.testing.assert_array_equal(combined >= .5, [True, True, False])
