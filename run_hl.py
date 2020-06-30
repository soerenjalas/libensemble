import subprocess
import socket
#what about mpi processor name - does that truncate?

def abbrev_nodenames(node_list):
    """Returns nodelist with only string upto first dot"""
    newlist = [s.split(".", 1)[0] for s in node_list]
    return newlist

print('\nRunning using hostlists')

buildstring = 'mpicc -o my_simtask.x libensemble/tests/unit_tests/simdir/my_simtask.c'
subprocess.check_call(buildstring.split())

hostname = socket.gethostname()
cmd='mpirun -hosts ' + hostname + ' -np 1 --ppn 1 ./my_simtask.x sleep 1'
print(cmd)
p = subprocess.Popen(cmd.split())
p.wait()

hostname = abbrev_nodenames([socket.gethostname()])[0]
cmd='mpirun -hosts ' + hostname + ' -np 1 --ppn 1 ./my_simtask.x sleep 1'
print(cmd)
p = subprocess.Popen(cmd.split())
p.wait()




