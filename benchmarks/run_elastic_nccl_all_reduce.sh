#!/bin/sh
set -e

cd $(dirname $0)
. ../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../mindspore)

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
    rm -fr logs
    trace kungfu_run 1 python3.7 ./elastic_gpu_all_reduce.py $(app_flags)
}

main
