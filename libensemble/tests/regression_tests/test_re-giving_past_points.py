# """
# Runs libEnsemble with generator function that is a greedy k-armed bandit
# method. This tests the re-giving of existing points in the history.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.k_armed_func import k_armed_func as sim_f
from libensemble.gen_funcs.persistent_k_armed_bandit import persistent_epsilon_greedy as gen_f
from libensemble.alloc_funcs.regiving_points import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = 4
num_arms = 10
draw_max = 200

np.random.seed(0)
sim_specs = {'sim_f': sim_f,
             'in': ['sim_id','arms'],
             'out': [('f', int)],
             'user': {'probabilities': np.random.uniform(0,1,num_arms)}
             }

gen_specs = {'gen_f': gen_f,
             'in': ['f'],
             'out': [('arms', float, n), ('f_results', int, draw_max), ('f_results_ind', int), ('estimated_p', float)],
             'user': {'epsilon': 0.1,
                      'init_pulls': 10,
                      'num_arms': 10,
                      'draw_max': draw_max,
                      'arm_dimension': n}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': draw_max, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    [_, counts] = np.unique(H['gen_time'], return_counts=True)
    print(counts)
    assert counts[0] == nworkers - 1, "The first gen_time should be common among gen_batch_size number of points"
    assert len(np.unique(counts)) > 1, "There is no variablitiy in the gen_times but there should be for the async case"

    save_libE_output(H, persis_info, __file__, nworkers)
