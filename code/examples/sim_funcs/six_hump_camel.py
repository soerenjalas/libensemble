from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import subprocess, os
import numpy as np

import time
from message_numbers import STOP_TAG


#Use float("inf") for no timeout - maybe that should be default
def poll_finish_or_kill(process, comm, timeout_sec=120.0, delay=3):
  start = time.time()
  status = MPI.Status()
  while time.time() - start < timeout_sec:
    time.sleep(delay)
    poll = process.poll()
    if poll is None:
        #Job still running - check for a kill signal  
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=STOP_TAG, status=status):   
            return False # Killed
    else:
        #Job complete
        return True
  raise RuntimeError("Process %s failed to complete in %d seconds" % (process,timeout_sec))
 

#Add poll to receive kill signal
def six_hump_camel_with_different_ranks_and_nodes(H, gen_info, sim_specs, libE_info):
    """
    Evaluates the six hump camel but also performs a system call (to show one
    way of evaluating a compiled simulation).
    """
    
    #SH Temp. to control output ***tofix 
    #Finding app helloworld.py will work if libensemble is installed.
    #Reconcile with sim_dir - but should copy original from either the
    #  installation or examples rather than writing in there
    
    app_location = os.path.abspath(os.path.join(os.path.dirname(__file__),"helloworld.py"))
    print("App location", app_location)
    
    #Create sim dir for the output for this job
    curr_dir = os.getcwd()
    if 'sim_dir_output' in sim_specs:
        output_dir_name = sim_specs['sim_dir_output']
        if not os.path.isdir(output_dir_name):
            try:
                os.mkdir(output_dir_name)
            except:
                raise("Cannot make simulation directory %s" % output_dir_name)
        try:
            os.chdir(output_dir_name)
        except:
            raise("Cannot cd to simulation directory %s" % output_dir_name)
    #print("Rank %d is in %s" % (MPI.COMM_WORLD.Get_rank(),output_dir_name))
    
    tag_out = None
    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):

        if 'blocking' in libE_info:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] +  list(libE_info['blocking'])
        else:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] 

        machinefilename = 'machinefile_for_sim_id=' + str(libE_info['H_rows'][i] )+ '_ranks='+'_'.join([str(r) for r in ranks_involved])

        with open(machinefilename,'w') as f:
            for rank in ranks_involved:
                b = sim_specs['nodelist'][rank] + '\n'
                f.write(b*H['ranks_per_node'][i])

        outfile_name = "outfile_"+ machinefilename+".txt"
        if os.path.isfile(outfile_name):
            os.remove(outfile_name)
        
        pause_time=1.0
        if 'pause_time' in sim_specs:
          pause_time=sim_specs['pause_time']    
                
        call_str = ["mpiexec","-np",str(H[i]['ranks_per_node']*len(ranks_involved)),"-machinefile",machinefilename,"python", app_location, str(pause_time)]
        
        #Original blocking call
        #process = subprocess.call(call_str, stdout = open(outfile_name,'w'), shell=False)
        
        #Call as non-blocking subprocess then poll for complete or kill signal
        ##***tofix: For kill should pass through comm - but for now use MPI.COMM_WORLD
        process = subprocess.Popen(call_str, stdout = open(outfile_name,'w'), shell=False)
        success = poll_finish_or_kill(process,MPI.COMM_WORLD)
        
        if success:
            print ("Completed job: %s" % (outfile_name))
        else:
            print ("Job not completed: %s -- Kill received" % (outfile_name))
            process.kill()
            tag_out = STOP_TAG
        

        O['f'][i] = six_hump_camel_func(x)

        # v = np.random.uniform(0,10)
        # print('About to sleep for :' + str(v))
        # time.sleep(v)
        
    #cd back to top dir
    if 'sim_dir_output' in sim_specs:
        os.chdir(curr_dir)  
        
    if tag_out == None:
        return O, gen_info
    else:
        return O, gen_info, tag_out



def six_hump_camel(H, gen_info, sim_specs, libE_info):
    """
    Evaluates the six_hump_camel_func and possible six_hump_camel_grad
    """
    del libE_info # Ignored parameter

    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):
        O['f'][i] = six_hump_camel_func(x)

        if 'grad' in O.dtype.names:
            O['grad'][i] = six_hump_camel_grad(x)

        if 'pause_time' in sim_specs:
            time.sleep(sim_specs['pause_time'])

    return O, gen_info


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2;
    term2 = x1*x2;
    term3 = (-4+4*x2**2) * x2**2;

    return  term1 + term2 + term3

def six_hump_camel_grad(x):
    """
    Definition of the six-hump camel gradient
    """

    x1 = x[0]
    x2 = x[1]
    grad = np.zeros(2)

    grad[0] = 2.0*(x1**5 - 4.2*x1**3 + 4.0*x1 + 0.5*x2)
    grad[1] = x1 + 16*x2**3 - 8*x2

    return grad

