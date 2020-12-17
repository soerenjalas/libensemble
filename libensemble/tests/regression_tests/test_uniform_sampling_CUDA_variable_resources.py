# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_uniform_sampling_CUDA_variable_resources.py
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi, local
# TESTSUITE_NPROCS: 2 4

import sys
import numpy as np
import pkg_resources

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_CUDA_variable_resources as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample_with_different_resources as gen_f
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor

nworkers, is_manager, libE_specs, _ = parse_args()

libE_specs['sim_dirs_make'] = True
libE_specs['ensemble_dir_path'] = './ensemble_CUDA_variable_w' + str(nworkers)

if libE_specs['comms'] == 'tcp':
    sys.exit("This test only runs with MPI or local -- aborting...")

# Get paths for applications to run
six_hump_camel_app = pkg_resources.resource_filename('libensemble.sim_funcs', 'six_hump_camel.py')

exctr = MPIExecutor()
exctr.register_calc(full_path=six_hump_camel_app, app_name='six_hump_camel')

n = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float)],
             'user': {}
             }

gen_specs = {'gen_f': gen_f,
             'in': ['sim_id'],
             'out': [('priority', float),
                     ('resource_sets', int),
                     ('x', float, n),
                     ('x_on_cube', float, n)],
             'user': {'initial_batch_size': 5,
                      'give_all_with_same_priority': False,  # SH TODO: Really an alloc option
                      'max_resource_sets': nworkers,
                      'lb': np.array([-3, -2]),
                      'ub': np.array([3, 2])}
             }

alloc_specs = {'alloc_f': give_sim_work_first,
               'out': [('allocated', bool)],
               'user': {'batch_mode': False,
                        'num_active_gens': 1}
               }

persis_info = add_unique_random_streams({}, nworkers + 1)
exit_criteria = {'sim_max': 40, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs, alloc_specs=alloc_specs)

if is_manager:
    assert flag == 0
    save_libE_output(H, persis_info, __file__, nworkers)
