Instructions for installing on Summit with Spack
================================================

The following instructions have been tested on the OLCF Summit supercomputer.

Setting up
----------

This can be run on a Summit login node. Follow the Linux install instructions
to get spack and set up a Spack environment.

At time of writing you will need a gcc that is more up to date than default::

    module load gcc/9.1.0
    spack compiler find

Install libEnsemble
-------------------

Then follow the Linux instructions to install ``py-libensemble``.
