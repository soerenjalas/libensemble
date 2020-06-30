import subprocess
import socket
#what about mpi processor name - does that truncate?

def abbrev_nodenames(node_list):
    """Returns nodelist with only string upto first dot"""
    newlist = [s.split(".", 1)[0] for s in node_list]
    return newlist

def write_mach(mfile):
    with open(mfile, 'w') as f:
        f.write(socket.gethostname() + '\n')

def print_mach(mfile):
    print('{} contains:'.format(mfile))
    with open(mfile, 'r') as f:
        print(f.read())


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
nodelist = abbrev_nodenames(nodelist)[0]
print('new nodelist:',nodelist)

mfile = 'mach2'
write_mach(mfile)
print_mach(mfile)
cmd='mpirun -machinefile mach2 -np 1 --ppn 1 ./my_simtask.x sleep 1'
print(cmd)
p = subprocess.Popen(cmd.split())
p.wait()





