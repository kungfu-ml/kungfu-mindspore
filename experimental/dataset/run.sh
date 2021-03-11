#!/bin/sh

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

app_flags() {
    echo --device GPU
    echo --dataset_path $HOME/var/data/cifar/cifar-10-batches-bin
    # echo --batch_size 100
    echo --batch_size 32
}

main() {
    rm -fr logs
    local np=1
    prun $np ./dataset-example.py $(app_flags)
}

main
