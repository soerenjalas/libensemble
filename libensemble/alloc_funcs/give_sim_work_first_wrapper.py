import numpy as np
# SH TODO:  Consider importing a class and using as object functions
from libensemble.tools.alloc_support import (sim_work, gen_work, count_gens,
                                             get_avail_workers_by_group, assign_workers)


# SH TODO: Either replace give_sim_work_first or add a different alloc func (or file?)
#          Check/update docstring
#          'worker' should maybe be 'workerID' so know its an int - not an object... (alt. wid or wrk_id etc)
def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most ``alloc_specs['user']['num_active_gens']``)
    generator instances.

    Allows for a ``alloc_specs['user']['batch_mode']`` where no generation
    work is given out unless all entries in ``H`` are returned.

    Allows for ``blocking`` of workers that are not active, for example, so
    their resources can be used for a different simulation evaluation.

    Can give points in highest priority, if ``'priority'`` is a field in ``H``.

    This is the default allocation function if one is not defined.

    .. seealso::
        `test_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling.py>`_ # noqa
    """

    Work = {}
    gen_count = count_gens(W)

    # SH TODO: Determine gen_specs/alloc_specs (could be object or dictionary).
    params = Params()
    params.gen_count = gen_count
    params.give_all_with_same_priority = gen_specs['user'].get('give_all_with_same_priority', False)
    params.max_active_gens = 1 #  alloc_specs['user'].num_active_gens
    params.batch_mode = alloc_specs['user'].get('batch_mode', False)
    params.sim_zrw = False
    params.gen_zrw = True if any_zero_resource_workers() else None  # True if they exist, else indifferent.
    #params.gen_zrw = any_zero_resource_workers()  # this should also work fine as if no zrw, its fine to look for False.

    Work, persis_info = _give_sim_work_first(W, H, sim_specs['in'], gen_specs['in'], persis_info, params)

    return Work, persis_info
