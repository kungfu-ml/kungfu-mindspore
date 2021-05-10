#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path $PWD/../../mindspore)
. ../../scripts/launcher.sh

cd $HOME/Desktop/mindspore/model_zoo/official/cv/resnet

train_flags() {
    echo --net=resnet50
    echo --dataset=cifar10
    echo --device_target=GPU
    echo --dataset_path=$HOME/var/data/cifar/cifar-10-batches-bin
}

eval_flags() {
    echo --net=resnet50
    echo --dataset=cifar10
    echo --device_target=GPU
    echo --dataset_path=$HOME/var/data/cifar/cifar-10-batches-bin
    echo --checkpoint_path resnet-5_250.ckpt
}

main() {
    # rm -fr cuda_meta_*
    # rm -fr resnet-graph.meta
    # srun train.py $(train_flags)
    srun eval.py $(eval_flags)
}

main
