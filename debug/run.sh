#!/bin/sh
set -e

cd $(dirname $0)
MS_ROOT=$PWD/../mindspore
. ../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../mindspore)
. ../scripts/launcher.sh

app_flags() {
    # echo --device CPU
    echo --device GPU
}

main() {
    rm -fr logs
    prun 2 ./debug.py $(app_flags)
}

main
