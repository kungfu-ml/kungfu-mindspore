#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)

app_flags() {
    echo --net=resnet50
    echo --dataset=cifar10
    echo --dataset_path=$HOME/var/data/cifar/cifar-10-batches-bin
    echo --device_target="GPU"
    # echo --device_num=4
    # echo --run_kungfu=True
    # echo --elastic=True
}

train() {
    rm -fr logs
    rm -fr resnet-graph.meta
    rm -fr ckpt*
    rm -fr cuda_meta*
    /usr/bin/python3.7 train.py $(app_flags)
}

train
