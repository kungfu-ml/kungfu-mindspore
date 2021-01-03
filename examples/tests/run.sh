#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../../mindspore

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs:$ROOT/build/mindspore/_deps/nccl-src/build/lib

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

test_broadcast_op() {
    local device=$1
    kungfu_run python3.7 test_broadcast_op.py --device $device --dtype i32
    kungfu_run python3.7 test_broadcast_op.py --device $device --dtype f32
}

test_allreduce_op() {
    local device=$1
    kungfu_run python3.7 test_allreduce_op.py --device $device --dtype i32
    kungfu_run python3.7 test_allreduce_op.py --device $device --dtype f32
}

test_all() {
    local device=$1
    test_broadcast_op $device
    test_allreduce_op $device
}

# test_all CPU
test_all GPU
