#!/bin/sh
set -e

cd $(dirname $0)

TAG=$(cat tag.txt)

cd mindspore
git checkout -f

cp -r ../ops mindspore/
cp -r ../ccsrc mindspore/
git apply ../patches/$TAG/a.patch
git diff
git status

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

./build.sh -e gpu
