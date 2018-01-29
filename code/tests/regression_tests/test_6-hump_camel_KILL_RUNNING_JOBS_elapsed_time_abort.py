# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html 
# 
# Execute via the following command:
#    mpiexec -np 4 python3 call_6-hump_camel_test_elapsed_time.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
sys.path.append('../../src')
from libE import libE

# Import sim_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs'))
from six_hump_camel import six_hump_camel_with_different_ranks_and_nodes

# Import gen_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from uniform_sampling import uniform_random_sample_with_different_nodes_and_ranks

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

import argparse
#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m','--mfile',action="store",dest='machinefile',
                    help='A machine file containing ordered list of nodes required for each libE rank')
args = parser.parse_args()

try:
    libE_machinefile = open(args.machinefile).read().splitlines()
except:
    if MPI.COMM_WORLD.Get_rank() == 0:        
        print("WARNING: No machine file provided - defaulting to local node")
    libE_machinefile = [MPI.Get_processor_name()]*MPI.COMM_WORLD.Get_size()

script_name = os.path.splitext(os.path.basename(__file__))[0]

#Change to app output dir
sim_dir_output_tmp = 'sim_dir_' + script_name # ***tofix: This is a massive kluge - to be reconciled with sim_dir ...

#if not os.path.isdir(output_dir_name):
#    os.mkdir(output_dir_name)
#os.chdir(output_dir_name)

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': [six_hump_camel_with_different_ranks_and_nodes], # This is the function whose output is being minimized
            'in': ['x','num_nodes','ranks_per_node'], # These keys will be given to the above function
            'out': [('f',float), # This is the output from the function being minimized
                    ],
             'nodelist': libE_machinefile,
             #'sim_dir': sim_dir_name,
             'sim_dir_output': sim_dir_output_tmp,
             'pause_time': 60,
             # 'save_every_k': 10
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample_with_different_nodes_and_ranks,
             'in': ['sim_id'],
             'out': [('x',float,2),
                     ('priority',float),
                     ('num_nodes',int),
                     ('ranks_per_node',int),
                    ],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'initial_batch_size': 5,
             'max_ranks_per_node': 8,
             'max_num_nodes': MPI.COMM_WORLD.Get_size()-1,
             'num_inst': 1,
             'batch_mode': False,
             'give_all_with_same_priority': True,
             # 'save_every_k': 10
             }

# Tell libEnsemble when to stop
exit_criteria = {'elapsed_wallclock_time': 3}

np.random.seed(1)

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    eprint(flag)
    eprint(H)
    assert flag == 2
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    # if flag == 2:
    #     print("\n\n\nKilling COMM_WORLD")
    #     MPI.COMM_WORLD.Abort()
