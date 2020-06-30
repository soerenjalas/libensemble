import subprocess
import socket
#what about mpi processor name - does that truncate?

def abbrev_nodenames(node_list):
    """Returns nodelist with only string upto first dot"""
    newlist = [s.split(".", 1)[0] for s in node_list]
    return newlist

def write_mach(mfile, hostname=socket.gethostname()):
    with open(mfile, 'w') as f:
        f.write(hostname + '\n')

def print_mach(mfile):
    print('{} contains:'.format(mfile))
    with open(mfile, 'r') as f:
        print(f.read())

buildstring = 'mpicc -o my_simtask.x libensemble/tests/unit_tests/simdir/my_simtask.c'
subprocess.check_call(buildstring.split())

# First machinefiles - then nodelists
mfile = 'mach'
write_mach(mfile)
print_mach(mfile)
cmd='mpirun -machinefile mach -np 1 --ppn 1 ./my_simtask.x sleep 1'
print(cmd)
p = subprocess.Popen(cmd.split())
p.wait()

nodelist = []
with open(mfile, 'r') as f:
    for line in f:
        nodelist.append(line.rstrip())
nodelist = abbrev_nodenames(nodelist)
print('new nodelist:',nodelist)

mfile = 'mach2'
#import pdb;pdb.set_trace()
write_mach(mfile, hostname=nodelist[0])
print_mach(mfile)
cmd='mpirun -machinefile mach2 -np 1 --ppn 1 ./my_simtask.x sleep 1'
print(cmd)
p = subprocess.Popen(cmd.split())
p.wait()

#Test definitly wrong
mfile = 'mach3'
write_mach(mfile, hostname='defwrong')

with open(mfile, 'w') as f:
    f.write('defwrong' + '\n')
print_mach(mfile)
cmd='mpirun -machinefile mach3 -np 1 --ppn 1 ./my_simtask.x sleep 1'
print(cmd)
p = subprocess.Popen(cmd.split())
p.wait()


