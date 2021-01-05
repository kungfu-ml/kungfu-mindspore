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

export KUNGFU_MINDSPORE_DEBUG=1
# export KUNGFU_CONFIG_LOG_LEVEL=1

kungfu_run_flags() {
    local np=$1

    echo -q
    echo -logdir logs
    echo -logfile kungfu-run.log
    echo -np $np

    echo -w
    local config_port=9999
    echo -builtin-config-port $config_port
    echo -config-server http://127.0.0.1:$config_port/config
}

kungfu_run() {
    local np=$1
    shift
    kungfu-run $(kungfu_run_flags $np) $@
}

app_flags() {
    # echo --device CPU
    echo --device GPU

    echo --steps 15

    # echo --model resnet50
    echo --model one
}

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    trace kungfu_run 1 python3.7 ./elastic_gpu_all_reduce.py $(app_flags)
}

rm -fr logs
# export GLOG_v=0
main
