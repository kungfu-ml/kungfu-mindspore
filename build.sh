#!/bin/sh
set -e

cd $(dirname $0)
. ./scripts/measure.sh

# TAG=$(cat tag.txt)
TAG=$(cat stable-tag.txt)
echo "using TAG=$TAG"

cd mindspore
git checkout -f

cp -r ../ops mindspore/
cp -r ../ccsrc mindspore/
git apply ../patches/$TAG/a.patch
# git diff
# git status

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc
export ENABLE_KUNGFU=ON

measure ./build.sh -e gpu
