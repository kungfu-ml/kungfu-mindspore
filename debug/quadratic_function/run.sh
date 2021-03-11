#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..
MS_ROOT=$ROOT/mindspore

. $ROOT/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path $MS_ROOT)
. $ROOT/scripts/launcher.sh

app_flags() {
    # echo --device CPU
    echo --device GPU

    echo --epochs 3
}

main() {
    rm -fr logs
    srun quadratic_function.py $(app_flags)
}

main
