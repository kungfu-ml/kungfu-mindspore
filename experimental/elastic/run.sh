#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -logfile kungfu-run.log
    echo -np 2
    # echo -np 1
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
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
    trace kungfu_run python3.7 ./optimizer.py $(app_flags)
}

rm -fr logs
# export GLOG_v=0
main
