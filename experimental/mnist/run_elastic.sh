#!/bin/sh

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

app_flags() {
    echo --device GPU
    echo --data-path $HOME/var/data/mindspore/mnist

    echo --save-ckpt
    echo --ckpt-dir checkpoint

    echo --run-train
    # echo --run-test

    # hyper parameters
    echo --batch-size 200
    echo --epochs 2
    echo --learning-rate 0.1
    echo --momentum 0.9

    # debug
    # echo --log-step
    echo --log-loss
}

main() {
    rm -fr logs
    rm -fr cuda_meta_*
    rm -fr analyze_fail.dat
    rm -fr *.ckpt
    rm -fr *.meta
    rm -fr checkpoint

    srun ./main_elastic.py $(app_flags) --use-kungfu
}

main
