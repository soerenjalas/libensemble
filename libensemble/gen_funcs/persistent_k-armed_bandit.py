import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg


def persistent_uniform(H, persis_info, gen_specs, libE_info):
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
    n = 4  # Dimension of each arm

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(num_arms, dtype=gen_specs['out'])
        H_o['arms'] = persis_info['rand_stream'].uniform(0, 1, (num_arms, n))
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            H_o['f'] = calc_in['f']

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
