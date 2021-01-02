#!/bin/sh
set -e

cp -r ops mindspore/mindspore/
cp -r ccsrc mindspore/mindspore/

cd mindspore

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

./build.sh -e gpu
