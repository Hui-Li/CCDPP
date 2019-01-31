#!/usr/bin/env bash

rm -rf build-ccdpp
mkdir build-ccdpp
cd build-ccdpp
cmake ../
make

data_folder="../../mf_data/netflix"
output_folder="../../output/netflix"
dimension="40"
lambda="0.05"
max_iter="5"
max_inner_iter="5"
thread=4
output=True

mkdir -p $output_folder

./runCCDPP --thread $thread --k $dimension --max_iter $max_iter --max_inner_iter $max_inner_iter --data_folder $data_folder --output_folder $output_folder --output $output
