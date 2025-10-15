#!/bin/bash

for npxy in 2 3 4 5 6 ; do
    nprocs=$((npxy*npxy))
    echo $nprocs
    cat par.nn.template | sed -e "s/__NPX__/$npxy/" | sed -e "s/__NPY__/$npxy/" > par.nn
    mpif90 advec_2d_par2_welton.f90 -o prg
    for t in 1 2 3 4 5 ; do
        mpiexec -np $nprocs ./prg >> ti$nprocs.txt
        sleep 2
    done

done
