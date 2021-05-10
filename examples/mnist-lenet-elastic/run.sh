#!/bin/sh
set -e

cd $(dirname $0)
ROOT=$PWD/../..

. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

train_flags() {
    local data_dir=$HOME/var/data/mindspore/mnist

    echo --device $(default_device)
    echo --data-dir $data_dir
    echo --batch-size 200

    echo --init-ckpt $ROOT/gist/mindspore-lenet-checkpoints/mnist-lenet-1618324040.ckpt

    # echo --epoch-size 1
    # echo --repeat-size 1
    # echo --run-test
}

kungfu_train() {
    rm -f *.meta
    erun 1 train.py $(train_flags) --use-kungfu
}

main() {
    rm -f *.meta
    rm -fr logs
    erun 1 train.py $(train_flags) --use-kungfu --use-kungfu-elastic
}

main
