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

# export KUNGFU_MINDSPORE_DEBUG=1

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -logfile kungfu-run.log
    echo -np $np
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
    # echo --device CPU
    echo --device GPU

    echo --warmup-steps 1
    echo --steps 8

    # echo --model empty
    # echo --model vgg16
    echo --model resnet50
    # echo --collective mindspore
    # echo --collective kungfu
}

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    # kungfu_run python3.7 ./hello_world.py $(app_flags)
    # for np in $(seq  4); do
    np=4
    trace kungfu_run python3.7 ./benchmark_all_reduce.py $(app_flags)
    # mpi_run python3.7 ./benchmark_all_reduce.py $(app_flags)
}

rm -fr logs
# export GLOG_v=0
main