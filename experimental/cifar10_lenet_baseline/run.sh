#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

# INIT_CKPT=1615216120
# INIT_CKPT_WITH_BN=1615220287
INIT_CKPT_WITH_BN2=1615550013

hardware_flags() {
    echo --device GPU
    echo --device-batch-size 200
    # echo --device-batch-size 100
    # echo --device-batch-size 50
}

config_flags() {
    echo --data-path $HOME/var/data/mindspore/cifar10

    echo --save-ckpt
    echo --ckpt-dir checkpoint

    # echo --init-ckpt seeds/cifar10-lenet-$INIT_CKPT.ckpt
    # echo --init-ckpt seeds/cifar10-lenet-bn-$INIT_CKPT_WITH_BN.ckpt
    echo --init-ckpt seeds/cifar10-lenet-bn2-$INIT_CKPT_WITH_BN2.ckpt
}

hyper_parameters() {
    echo --logical-batch-size 200
    echo --epochs 10

    # echo --optimizer sgd
    echo --optimizer momentum
    echo --learning-rate 0.01
    echo --momentum 0.9

    echo --ckpt-period 50
    # echo --ckpt-period 50
    # echo --ckpt-period 1 # for debug

    echo --use-bn
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
    # echo --stop-logical-step 250
    # echo --stop-logical-step 10 # for
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

    # srun ./main.py $(train_flags)

    # prun 1 ./main.py $(train_flags) --use-kungfu
    # prun 2 ./main.py $(train_flags) --use-kungfu
    prun 4 ./main.py $(train_flags) --use-kungfu
}

run_test() {
    # checkpoints_files=$(comma_join $(ls checkpoint/*.ckpt))
    # srun ./main.py $(test_flags)  --ckpt-files $checkpoints_files

    # prun 1 ./main.py $(test_flags) --use-kungfu
    # prun 2 ./main.py $(test_flags) --use-kungfu
    prun 4 ./main.py $(test_flags) --use-kungfu
}

summary() {
    du -hs checkpoint
    # md5sum checkpoint/*.npz >finger-print.$INIT_CKPT.txt
    # md5sum batch*.npz >data.md5.txt
}

main() {
    # cleanup
    run_train
    # run_test
    # summary
}

# trace run_init
trace main
# run_test
