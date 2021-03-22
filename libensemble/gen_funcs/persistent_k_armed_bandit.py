import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg


def update_local_history(calc_in, H):
    for i, sim_id in enumerate(calc_in['sim_id']):
        new = H[sim_id]['num_new_pulls']
        start = H[sim_id]['num_completed_pulls']
        end = start + new
        H['f_results'][sim_id][start:end] = calc_in['last_f_results'][i, :new]
        H['estimated_p'][sim_id] = np.mean(H[sim_id]['f_results'][:end])
        H['num_completed_pulls'][sim_id] = end
    print(H['num_completed_pulls'])


def persistent_epsilon_greedy(H, persis_info, gen_specs, libE_info):
    """
    This persistent generation function implements an epsilon-greedy strategy
    for a k-armed bandit problem. The generation function
    - generates the "num_arms" arms
    - intially: requests each arm to be evaluated "init_pulls" times
    - thereafter: produces a "batch_size" number of points. Each member of the batch is
      the best arm with with probability 1-"epsilon" and a random arm with probability "epsilon"

    .. seealso::
        `test_persistent_k-armed_bandit.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_k-armed_bandid.py>`_ # noqa
    """
    epsilon = gen_specs['user']['epsilon']
    init_pulls = gen_specs['user']['init_pulls']
    batch_size = gen_specs['user']['batch_size']
    num_arms = gen_specs['user']['num_arms']
    n = gen_specs['user']['arm_dimension']

    local_H = np.zeros(num_arms, dtype=gen_specs['out'])
    local_H['arms'] = persis_info['rand_stream'].uniform(0, 1, (num_arms, n))
    local_H['sim_id'] = range(num_arms)
    local_H['num_new_pulls'] = init_pulls

    tag = None

    # first send back all arms
    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], local_H[['sim_id', 'arms', 'num_new_pulls']])

    # Send batches until manager sends stop tag
    while tag not in [STOP_TAG, PERSIS_STOP]:
        update_local_history(calc_in, local_H)

        # Reset number of new pulls after updating local history
        local_H['num_new_pulls'] = 0
        best_arm = np.argmax(local_H['estimated_p'])
        for i in range(batch_size):
            if np.random.uniform() < epsilon:
                random_arm = np.random.choice(num_arms)
                local_H[random_arm]['num_new_pulls'] += 1
            else:
                local_H[best_arm]['num_new_pulls'] += 1

        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], local_H[['sim_id', 'arms', 'num_new_pulls']])

        break

    update_local_history(calc_in, local_H)

    return local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG
