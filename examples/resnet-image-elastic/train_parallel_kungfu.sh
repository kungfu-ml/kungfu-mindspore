#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
    echo --net=resnet50
    echo --dataset=cifar10
    echo --dataset_path=$HOME/var/data/mindspore/image/train/imagenet
    echo --device_num=4
    echo --device_target="GPU"
    echo --run_kungfu=True
}

train() {
    rm -fr resnet-graph.meta
    rm -fr ckpt_*
    rm -fr cuda_meta_*
    rm -fr logs
    kungfu_run \
        python3.7 train.py $(app_flags)
}

# export GLOG_v=3 # ERROR
# export GLOG_v=2 # WARNING
# export GLOG_v=1 # INFO
# export GLOG_v=0 # DEBUG

train
