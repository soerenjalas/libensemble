# Compare with using object for alloc_support
import numpy as np
from _give_sim_work_first import Params
from libensemble.tools.alloc_support import (avail_worker_ids,
                                             get_evaluated_points,
                                             sim_work, gen_work,
                                             count_persis_gens,
                                             get_avail_workers_by_group,
                                             assign_workers,
                                             any_zero_resource_workers())


def only_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
    """

    Work = {}
    gen_count = count_persis_gens(W)
    async_return = gen_specs['user'].get('async', False)

    #some variation of... (*what if using priorities? - that maybe why I did based on sim_id/index).
    if sum(H['returned']) < alloc_specs['user']['initial_sample_size']:
        async_return = False

    # This should be gen_specs['in'] in my opinion (or gen_specs['update'] ???)
    gen_in = sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')]

    if persis_info.get('gen_started') and gen_count == 0:
        # The one persistent gen is done. Exiting
        return Work, persis_info, 1

    # Give evaluated results back to a running persistent gen
    for i in avail_worker_ids(W, persistent=True):
        if async_return or all_returned(H):
            points_evaluated = get_evaluated_points(H)
            if np.any(points_evaluated):
                Work[i] = gen_work(i, gen_in, np.atleast_1d(points_evaluated),
                                persis_info[i], persistent=True)
                H['given_back'][points_evaluated] = True

    # SH TODO: Determine gen_specs/alloc_specs (could be object or dictionary).
    params = Params()
    params.gen_count = gen_count
    params.give_all_with_same_priority = gen_specs['user'].get('give_all_with_same_priority', False)
    params.max_active_gens = 1 #  alloc_specs['user'].num_active_gens
    params.batch_mode = alloc_specs['user'].get('batch_mode', False)
    params.sim_zrw = False
    params.gen_zrw = True if any_zero_resource_workers() else None  # True if they exist, else indifferent.
    #params.gen_zrw = any_zero_resource_workers()  # this should also work fine as if no zrw, its fine to look for False.

    Work, persis_info = _give_sim_work_first(W, H, Work, sim_specs['in'], gen_specs['in'], persis_info, params)

    return Work, persis_info, 0

