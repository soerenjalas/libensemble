#!/usr/bin/env python
# Test of executor module for libensemble
# Test does not require running full libensemble
import os
from libensemble.executors.executor import Executor


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring = 'mpicc -o my_simjob.x simdir/my_simjob.c'
    # subprocess.run(buildstring.split(),check=True) # Python3.5+
    subprocess.check_call(buildstring.split())

# --------------- Calling script ------------------------------------------


# sim_app = 'simdir/my_simjob.x'
# gen_app = 'gendir/my_gentask.x'

# temp
sim_app = './my_simjob.x'

if not os.path.isfile(sim_app):
    build_simfunc()

USE_BALSAM = False  # Take as arg
# USE_BALSAM = True # Take as arg

# Create and add exes to registry
if USE_BALSAM:
    from libensemble.executors.balsam_executor import Balsam_MPI_Executor
    exctr = Balsam_MPI_Executor()
else:
    from libensemble.executors.mpi_executor import MPI_Executor
    exctr = MPI_Executor()

exctr.register_calc(full_path=sim_app, calc_type='sim')

# Alternative to IF could be using eg. fstring to specify: e.g:
# EXECUTOR = 'Balsam'
# registry = f"{EXECUTOR}Register()"


# --------------- Worker: sim func ----------------------------------------
# Should work with Balsam or not

def polling_loop(exctr, task, timeout_sec=20.0, delay=2.0):
    import time
    start = time.time()

    while time.time() - start < timeout_sec:
        time.sleep(delay)
        print('Polling at time', time.time() - start)
        task.poll()
        if task.finished:
            break
        elif task.state == 'WAITING':
            print('Task waiting to launch')
        elif task.state == 'RUNNING':
            print('Task still running ....')

        # Check output file for error
        if task.stdout_exists():
            if 'Error' in task.read_stdout():
                print("Found (deliberate) Error in ouput file - cancelling task")
                exctr.kill(task)
                time.sleep(delay)  # Give time for kill
                break

    if task.finished:
        if task.state == 'FINISHED':
            print('Task finished succesfully. Status:', task.state)
        elif task.state == 'FAILED':
            print('Task failed. Status:', task.state)
        elif task.state == 'USER_KILLED':
            print('Task has been killed. Status:', task.state)
        else:
            print('Task status:', task.state)
    else:
        print("Task timed out")
        exctr.kill(task)
        if task.finished:
            print('Now killed')
            # double check
            task.poll()
            print('Task state is', task.state)


# Tests

# From worker call Executor by different name to ensure
# getting registered app from Executor
exctr = Executor.executor

print('\nTest 1 - should complete succesfully with status FINISHED :\n')
cores = 4
args_for_sim = 'sleep 5'

task = exctr.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
polling_loop(exctr, task)

print('\nTest 2 - Task should be USER_KILLED \n')
cores = 4
args_for_sim = 'sleep 5 Error'

task = exctr.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
polling_loop(exctr, task)
