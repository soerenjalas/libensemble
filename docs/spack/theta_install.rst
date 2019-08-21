Building libEnsemble with Spack on Theta.
=========================================

The following instructions have been tested on the ALCF Theta supercomputer.

To install libensemble through Spack on Theta:
A working method is to fetch packages on the front-end nodes and then install via the MOM nodes in
an interactive session. It is recommended to use spack environments, though not essential.

Setting up
----------

Follow the Linux install instructions to get spack and set up a Spack environment.

Fetch dependencies on the front-end login nodes::

    spack fetch --dependencies py-libensemble

This line should include any of the options you will give on the install line following py-libensemble.


Install libEnsemble
-------------------

Now get an interative session for the installation. E.g::

    qsub -A <projectID> -n 1 -q debug-flat-quad -t 60 -I
    
Note that 60 mins is the longest you can use in debug queues at time of writing. If the build times out,
it can be restarted by using the ``--dont-restage`` flag in the build (see below).

In the interactive session, make sure your spack environment is set up and activated. Install with::

    aprun -cc none -n 1 python /home/shudson/spackdev/bin/spack install --dont-restage -j 16 py-libensemble
    
Again adding any install options at the end (as in the Linux instructions).
