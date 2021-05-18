#!/bin/sh
set -e

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 1

    local enable_elastic=1
    if [ $enable_elastic -eq 1 ]; then
        echo -w
        echo -builtin-config-port 9100
        echo -config-server http://127.0.0.1:9100/config
    fi
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
    echo --net=resnet50
    echo --dataset=imagenet2012
    echo --dataset_path=$HOME/var/data/mindspore/image/train/
    echo --device_num=4
    echo --device_target="GPU"
    echo --run_kungfu=True
    echo --elastic=True
}

train() {
    rm -fr logs
    rm -fr resnet-graph.meta
    rm -fr ckpt_*
    rm -fr cuda_meta_*
    kungfu_run \
        python3.7 train.py $(app_flags)
}

train
