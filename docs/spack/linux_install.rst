Instructions for installing with Spack
======================================

The following instructions have been tested on a Linux Ubuntu 18.04 platfrom.


Setting up
----------

Clone and setup spack::

    git clone git@github.com:spack/spack.git
    cd spack
    export SPACK_ROOT=$PWD    
    export PATH=$SPACK_ROOT/bin:$PATH

Create a Spack environment to isolate work (optional)::

    . $SPACK_ROOT/share/spack/setup-env.sh  # Set up environment tools
    export MYENV=libe_0-5-2                 # Name your environment
    spack env create $MYENV
    spack env activate -p --with-view $MYENV
    spack compiler find
    
On the activate line the ``-p`` prepends you path with environment name. The ``--with-view`` will make sure you
pick up the spack installed packages when you run the code.

Check the spec. This will show you exactly what Spack will install. For default build configuration::

    spack spec py-libensemble

libEnsemble can be built with different versions and dependency variants. Variants are installed by
default for some versions. For example, MPI is required before version 0.5.0 and so it is a default 
dependency. From version 0.5.0 it is optional (as libEnsemble has alternative communications). For example,
to spec the version 0.5.2 with the dependencies required for the full libEnsemble testsuite::

    spack spec py-libensemble @0.5.2 +mpi +scipy +petsc4py +nlopt

This may take some time. Using the install option ``--dont-restage`` will make sure repeated sessions
will not delete previous state of install.


Install libEnsemble
-------------------

Run the install with all the same arguments as your spec line. E.g. For default options::

    spack install py-libensemble

Install version 0.5.2 with the dependencies required for the full libEnsemble testsuite::

    spack install py-libensemble @0.5.2 +mpi +scipy +petsc4py +nlopt


Now use the package (while in environment)
------------------------------------------

*Note* This will work if you specified ``--with-view`` when activating the environment::

    $ python
    ....
    >>> import libensemble
    >>> libesnemble.__path__
    >>> libensemble.__version__

These should show the path to installed libEnsemble and the correct version number.

Deactiate::

    spack env deactivate

OR::

    despacktivate


Notes on using environment
--------------------------

To see list of environments::

    spack env list

To check currently loaded environment::

    spack env status
