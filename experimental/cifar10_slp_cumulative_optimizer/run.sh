#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

INIT_CKPT=1615125968

hardware_flags() {
    echo --device GPU

    # echo --device-batch-size 200
    # echo --device-batch-size 100
    echo --device-batch-size 50
}

config_flags() {
    echo --data-path $HOME/var/data/mindspore/cifar10

    echo --save-ckpt
    echo --ckpt-dir checkpoint

    echo --init-ckpt seeds/cifar10-slp-$INIT_CKPT.ckpt
}

hyper_parameters() {
    #
    echo --logical-batch-size 200 # 300 logical steps (60000 / 200)
    echo --epochs 2

    # echo --optimizer momentum
    echo --optimizer sgd
    echo --learning-rate 0.01
    # echo --momentum 0.9

    echo --ckpt-period 10
}

init_flags() {
    hardware_flags
    config_flags
    hyper_parameters

    echo --mode init
}

train_flags() {
    hardware_flags
    config_flags
    hyper_parameters

    echo --mode train

    # debug parameters
    # echo --log-step
    echo --log-loss

    # echo --stop-logical-step 10
}

test_flags() {
    hardware_flags
    config_flags
    hyper_parameters

    echo --test-batch-size 200
    echo --mode test
}

comma_join() {
    local IFS=","
    echo "$*"
}

cleanup() {
    rm -fr __pycache__
    rm -fr *.ckpt
    rm -fr *.data
    rm -fr *.meta
    rm -fr analyze_fail.dat
    rm -fr checkpoint
    rm -fr cuda_meta_*
    rm -fr logs
}

run_init() {
    cleanup
    srun ./main.py $(init_flags)
}

run_train() {
    mkdir -p checkpoint
    # get_logs "train" \
    # srun
    prun 1 ./main.py $(train_flags)
}

run_test() {
    checkpoints_files=$(comma_join $(ls checkpoint/*.ckpt))
    # get_logs "test" \
    srun ./main.py $(test_flags) --ckpt-files $checkpoints_files
}

summary() {
    du -hs checkpoint
    # md5sum checkpoint/*.npz >finger-print.$INIT_CKPT.txt
    # md5sum batch*.npz >data.md5.txt
    # md5sum checkpoint/*device* >tmp.txt
}

run_plot() {
    # ./plot.py
    # mkdir -p plot
    # cat -n test-result.txt | awk '{print $1, $4}' >plot/data.txt
    pdflatex plot.tex
}

main() {
    cleanup
    run_train
    run_test
    summary
}

# trace run_init
trace main
# run_test
# run_plot
