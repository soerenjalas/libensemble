import numpy as np
import sys
from libensemble.tools.alloc_support import (avail_worker_ids,
                                             sim_work, gen_work,
                                             count_persis_gens,
                                             get_avail_workers_by_group,
                                             assign_workers)


class Params:
    """Config params - simply nicer than dictionary"""
    def __init__():
        self.gen_count = 0
        self.give_all_with_same_priority = False
        self.max_active_gens = sys.maxsize - 1
        self.batch_mode = False
        #self.gen_persis = None  # always put "new" gens on non-persistent?
        #self.sim_persis = None  # may sometime want to put sims on persistent?
        self.gen_zrw = None  # True/False/None
        self.sim_zrw = None  # True/False/None


#def _give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
def _give_sim_work_first(W, H, Work, sim_in, gen_in, persis_info, params):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
    """

# Differences
    #Can you have persis and non-persis gens in same ensemble (or just persis_gen option req)
    # 1
    # gen_count = count_gens(W)
    # gen_count = count_persis_gens(W)

    # 2
    # avail_workers = get_avail_workers_by_group(W)
    # avail_workers = get_avail_workers_by_group(W, persistent=False, zero_resource_workers=False)
    #
    # Need:  avail_workers = get_avail_workers_by_group(W, persistent=sim_peris, zero_resource_workers=sims_zeros)
    # gen_workers/sim_workers options (each with different resource sizes), might make easier?

    # 3
    # give_sim_work_first - has alloc_specs['user']['num_active_gens'] (now use alloc_specs!)
    # I think both can use this - but maybe default to one!
    # Though I guess - persis case - assumes only one with the exit condition (removed from here).

    #Work = {}
    #gen_count = count_persis_gens(W)

    points_to_evaluate = get_points_to_evaluate(H)
    #task_avail = ~H['given']  # SH TODO: Unchanged - but what if allocated space in array that is not genned?

    # Dictionary of workers by group (node).
    #avail_workers = get_avail_workers_by_group(W, persistent=False, zero_resource_workers=False)
    avail_workers = get_avail_workers_by_group(W, persistent=False, zero_resource_workers=params.sim_zrw)
    # print('avail_workers for sim',avail_workers)

    while any(avail_workers.values()):

        if not np.any(points_to_evaluate):
            break

        # Perform sim evaluations (if they exist in History).
        sim_ids_to_send = np.nonzero(points_to_evaluate)[0][0]  # oldest point

        nworkers_req = (np.max(H[sim_ids_to_send]['resource_sets'])
                        if 'resource_sets' in H.dtype.names else 1)

        # If more than one group (node) required, allocates whole nodes - also removes from avail_workers
        worker_team = assign_workers(avail_workers, nworkers_req)
        # print('AFTER ASSIGN sim ({}): avail_workers: {}'.format(worker_team,avail_workers), flush=True)

        if not worker_team:
            break  # No slot found - insufficient available resources for this work unit

        workerID = worker_team[0]
        #atleast_1d should prob go in sim_work (or pack_sim_work) function.
        #sim_work(Work, workerID, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info[workerID])
        Work[workerID] = sim_work(workerID, sim_in, np.atleast_1d(sim_ids_to_send), persis_info[workerID])

        points_to_evaluate[sim_ids_to_send] = False
        if len(worker_team) > 1:
            Work[workerID]['libE_info']['blocking'] = worker_team[1:]  # SH TODO: Maybe do in sim_work?

    # A separate loop/section as now need zero_resource_workers for gen.
    if not np.any(points_to_evaluate):
        #avail_workers = get_avail_workers_by_group(W, persistent=False, zero_resource_workers=True)
        avail_workers = get_avail_workers_by_group(W, persistent=False, zero_resource_workers=params.gen_zrw)
        # print('avail_workers for gen',avail_workers)

        while any(avail_workers.values()):
            # SH TODO: So we don't really need a loop here for this, but general case would allow multiple gens
            if gen_count == 0:
                # Finally, call a persistent generator as there is nothing else to do.
                gen_count += 1

                worker_team = assign_workers(avail_workers, 1)  # Returns a list even though one element
                if not worker_team:
                    break
                workerID = worker_team[0]
                # print('AFTER ASSIGN gen ({}): avail_workers: {}'.format(workerID,avail_workers), flush=True)
                gen_work(Work, workerID, gen_specs['in'], range(len(H)), persis_info[workerID],
                         persistent=True)
                persis_info['gen_started'] = True

    #return Work, persis_info, 0
    return Work, persis_info
