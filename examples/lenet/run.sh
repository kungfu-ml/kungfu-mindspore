#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

train_flags() {
    echo --data-dir $HOME/var/data/mindspore/mnist

    # echo --device CPU
    echo --device GPU

    # echo --epoch-size 1
    # echo --repeat-size 1
}

single_train() {
    rm -f *.meta
    python3.7 train.py $(train_flags)
}

kungfu_train() {
    rm -f *.meta
    rm -fr logs
    prun 1 train.py $(train_flags) --use-kungfu
}

main() {
    if [ $(hostname) = "platypus2" ]; then
        kungfu_train
    else
        single_train
    fi
}

main
