#!/usr/bin/env bash

rm -rf build-ccdpp-mpi
mkdir build-ccdpp-mpi
cd build-ccdpp-mpi
cmake ../
make

data_folder="../../../mf_data/netflix"
output_folder="../../output/netflix"
dimension="40"
lambda="0.05"
max_iter="5"
max_inner_iter="5"
thread=4
node=1
output=True

mkdir -p $output_folder

./runCCDPP_MPI --thread $thread --node $node --k $dimension --max_iter $max_iter --max_inner_iter $max_inner_iter --data_folder $data_folder --output_folder $output_folder --output $output