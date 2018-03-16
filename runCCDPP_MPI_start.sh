#!/usr/bin/env bash

# change the value in runCCDPP_MPI.sh accordingly
node=1

mpiexec -f "hosts" -n $node ./runCCDPP_MPI.sh