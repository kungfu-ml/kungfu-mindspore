#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../mindspore

KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs:$ROOT/build/mindspore/_deps/nccl-src/build/lib

mpi_run_flags() {
    echo -np 2
}

mpi_run() {
    echo "$@"
    mpirun $(mpi_run_flags) $@
    echo
}

test_broadcast_op() {
    local device=$1
    mpi_run python3.7 test_official_collective_ops.py --device $device --dtype i32 --op broadcast
    mpi_run python3.7 test_official_collective_ops.py --device $device --dtype f32 --op broadcast
}

test_allreduce_op() {
    local device=$1
    mpi_run python3.7 test_official_collective_ops.py --device $device --dtype i32 --op all_reduce
    mpi_run python3.7 test_official_collective_ops.py --device $device --dtype f32 --op all_reduce
}

test_all() {
    local device=$1
    test_allreduce_op $device
    # test_broadcast_op $device
}

# test_all CPU # Device target CPU is not supported.
test_all GPU
