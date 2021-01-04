#!/bin/sh
set -e

join_path() {
    local IFS=":"
    echo "$*"
}

cd $(dirname $0)
ROOT=$PWD/../mindspore

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

ld_library_path() {
    echo $KUNGFU_LIB_PATH
    echo $ROOT/mindspore/lib
    echo $ROOT/build/mindspore/_deps/ompi-src/ompi/.libs
}

export LD_LIBRARY_PATH=$(join_path $(ld_library_path))

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
    echo
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

test_all CPU
test_all GPU
