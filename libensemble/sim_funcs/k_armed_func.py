"""
This module contains an example k-armed bandit evaluator
"""
__all__ = ['k_armed_func']

import numpy as np


def k_armed_func(H, persis_info, sim_specs, _):
    """
    Returns a 1 with probability np.random.uniform() as defined by np.seed(sim_id)

    .. seealso::
        `test_re-giving_past_points.py test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_re-giving_past_points.py>`_ # noqa
    """

    H_o = np.zeros(1, dtype=sim_specs['out'])

    assert len(H) == 1, "This function is only for single rows"

    p = sim_specs['user']['probabilities'][H['sim_id']]
    n = H['num_new_pulls'][0]

    H_o['last_f_results'][0, :n] = np.random.binomial(1, p=p, size=n)

    return H_o, persis_info
