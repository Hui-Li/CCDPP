#!/usr/bin/env bash

rm -rf build-ccdpp
mkdir build-ccdpp
cd build-ccdpp
cmake ../
make

data_folder="../../mf_data/netflix"

dimension="40"
lambda="0.05"
max_iter="5"
max_inner_iter="5"
thread=4

./runCCDPP --thread $thread --k $dimension --max_iter $max_iter --max_inner_iter $max_inner_iter --data_folder $data_folder
