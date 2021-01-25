#!/bin/sh
set -e

cd $(dirname $0)
MS_ROOT=$PWD/../mindspore

. ../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../mindspore)

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -logfile kungfu-run.log
    echo -np 2
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
    # echo --device CPU
    echo --device GPU
}

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    rm -fr logs
    # trace kungfu_run python3.7 ./debug.py $(app_flags)
    export NCCL_DEBUG=INFO
    trace kungfu_run $MS_ROOT/third_party/kungfu/bin/kungfu_debug_nccl
}

main
