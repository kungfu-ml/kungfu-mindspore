#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

INIT_CKPT=1613876569

hardware_flags() {
    echo --device GPU
    # echo --device-batch-size 200
    # echo --device-batch-size 100
    echo --device-batch-size 50
}

config_flags() {
    echo --data-path $HOME/var/data/mindspore/mnist
    echo --save-ckpt
    echo --ckpt-dir checkpoint

    echo --init-ckpt seeds/$INIT_CKPT.ckpt
}

hyper_parameters() {
    echo --logical-batch-size 200 # can converge
    # echo --logical-batch-size 400 # not converge
    # echo --logical-batch-size 500 # not converge
    echo --epochs 3

    # https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/lenet/src/config.py
    echo --optimizer momentum
    echo --learning-rate 0.01
    echo --momentum 0.9

    echo --ckpt-period 1
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
    echo --stop-logical-step 1
}

test_flags() {
    hardware_flags
    config_flags
    hyper_parameters

    echo --mode test
}

comma_join() {
    local IFS=","
    echo "$*"
}

cleanup() {
    rm -fr logs
    rm -fr cuda_meta_*
    rm -fr analyze_fail.dat
    rm -fr *.ckpt
    rm -fr *.meta
    rm -fr checkpoint
    rm -fr batch-*.npz
}

run_init() {
    cleanup
    srun ./main.py $(init_flags)
}

run_train() {
    mkdir -p checkpoint
    mkdir -p plot

    # get_logs "train" \
    srun ./main.py $(train_flags)
}

run_test() {
    checkpoints_files=$(comma_join $(ls checkpoint/*.ckpt))
    # get_logs "test" \
    srun ./main.py $(test_flags) --ckpt-files $checkpoints_files
}

summary() {
    du -hs checkpoint
    md5sum checkpoint/*.npz >finger-print.$INIT_CKPT.txt
    # md5sum batch*.npz >data.md5.txt
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
