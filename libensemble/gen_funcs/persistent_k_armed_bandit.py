import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg


def persistent_epsilon_greedy(H, persis_info, gen_specs, libE_info):
    """
    This persistent generation function implements an epsilon-greedy strategy
    for a k-armed bandit problem. The generation function 
    - generates the "num_arms" arms
    - intially: requests each arm to be evaluated "init_pulls" times 
    - thereafter: requests the best arm to be evaluated with probability
      1-"epsilon" and a random arm with probability "epsilon" 

    .. seealso::
        `test_persistent_k-armed_bandit.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_k-armed_bandid.py>`_ # noqa
    """
    epsilon = gen_specs['user']['epsilon']
    init_pulls = gen_specs['user']['init_pulls']
    num_arms = gen_specs['user']['num_arms']
    n = gen_specs['user']['arm_dimension']
    draw_max = gen_specs['user']['draw_max']

    import ipdb; ipdb.set_trace()

    dtype_list = [('arms', float, n), ('f_results', int, draw_max), ('f_results_ind', int), ('estimated_p', float)]

    H_o = np.zeros(num_arms, dtype=dtype_list)
    H_o['arms'] = persis_info['rand_stream'].uniform(0, 1, (num_arms, n))

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        for i in range(init_pulls):
            tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o['arms'])

            if calc_in is not None:
                H_o['f'] = calc_in['f']

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
