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
num_arms = 8
max_gen_calls = 10
init_pulls = 10
batch_size = 20
draw_max = init_pulls + (max_gen_calls-1)*batch_size

k = np.zeros(num_arms) 
k[0] = 1

np.random.seed(0)
sim_specs = {'sim_f': sim_f,
             'in': ['sim_id', 'arms', 'num_new_pulls'],
             'out': [('last_f_results', int, draw_max)],
             'user': {'probabilities': np.random.uniform(0, 1, num_arms)}
             }

gen_specs = {'gen_f': gen_f,
             'in': ['last_f_results', 'sim_id'],
             'out': [('sim_id', int), ('arms', float, n), ('num_new_pulls', int),
                     ('num_completed_pulls', int), ('f_results', int, draw_max), ('estimated_p', float)],
             'user': {'epsilon': 0.1,
                      'init_pulls': init_pulls,
                      'batch_size': batch_size,
                      'num_arms': num_arms,
                      'draw_max': draw_max,
                      'arm_dimension': n}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'gen_return_max': max_gen_calls}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    assert sum(H['num_completed_pulls']) > 0, "Why no completed pulls"
    save_libE_output(H, persis_info, __file__, nworkers)
