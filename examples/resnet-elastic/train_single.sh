#!/bin/sh
set -e

join_path() {
    local IFS=":"
    echo "$*"
}

cd $(dirname $0)
ROOT=$PWD/../../mindspore

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

ld_library_path() {
    echo $KUNGFU_LIB_PATH
    echo $ROOT/mindspore/lib
    echo $ROOT/build/mindspore/_deps/ompi-src/ompi/.libs
    echo $ROOT/build/mindspore/_deps/nccl-src/build/lib
}

export LD_LIBRARY_PATH=$(join_path $(ld_library_path))

app_flags() {
    echo --net=resnet50
    echo --dataset=cifar10
    echo --dataset_path=$HOME/var/data/cifar/cifar-10-batches-bin
    echo --device_target="GPU"
    # echo --device_num=4
    # echo --run_kungfu=True
    # echo --elastic=True
}

train() {
    rm -fr logs
    rm -fr resnet-graph.meta
    rm -fr ckpt*
    rm -fr cuda_meta*
    /usr/bin/python3.7 train.py $(app_flags) >out.log 2>err.log
}

train
