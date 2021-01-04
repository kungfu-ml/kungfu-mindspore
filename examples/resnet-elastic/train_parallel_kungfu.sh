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

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
    echo --net=resnet50
    echo --dataset=cifar10
    echo --dataset_path=$HOME/var/data/cifar/cifar-10-batches-bin
    echo --device_num=4
    echo --device_target="GPU"
    echo --run_kungfu=True
}

train() {
    rm -fr resnet-graph.meta
    rm -fr ckpt_*
    rm -fr cuda_meta_*
    kungfu_run \
        /usr/bin/python3.7 train.py $(app_flags)
}

# export GLOG_v=3 # ERROR
# export GLOG_v=2 # WARNING
# export GLOG_v=1 # INFO
# export GLOG_v=0 # DEBUG

train
