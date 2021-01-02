#!/bin/sh
set -e

if [ ! -d mindspore ]; then
    git clone https://gitee.com/mindspore/mindspore.git
fi
cd mindspore
git checkout $(cat ../tag.txt)

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

./build.sh -e gpu
