#!/bin/sh
set -e

cd $(dirname $0)
. ./scripts/measure.sh

TAG=$(cat tag.txt)
echo "using TAG=$TAG"

measure ./install-kungfu.sh

cd mindspore
git checkout -f

rm -fr mindspore/ccsrc/backend/kernel_compiler/cpu/kungfu
rm -fr mindspore/ccsrc/backend/kernel_compiler/gpu/kungfu
cp -r ../extension/* mindspore/

git apply ../patches/$TAG/prebuild/*.patch
git apply ../patches/$TAG/*.patch

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc
export ENABLE_KUNGFU=ON

measure ./build.sh -e gpu
