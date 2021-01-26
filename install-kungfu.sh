#!/bin/sh
set -e

GIT_URL=https://github.com/lsds/KungFu.git
GIT_TAG=v0.2.3
# GIT_URL=git@ssh.dev.azure.com:v3/lg4869/kungfu/kungfu
# GIT_TAG=ms-dev

cd $(dirname $0)/mindspore
MINDSPORE_ROOT=$PWD

PREFIX=$MINDSPORE_ROOT/third_party/kungfu

if [ ! -d KungFu ]; then
    git clone $GIT_URL KungFu
fi

cd KungFu
git fetch --tags
git checkout $GIT_TAG
# git pull

config_flags() {
    echo --prefix=$PREFIX
    echo --enable-nccl
    echo --with-nccl=$MINDSPORE_ROOT/build/mindspore/_deps/nccl-src/build
}

./configure $(config_flags)
make -j 8

if [ -d $PREFIX ]; then
    rm -fr $PREFIX
fi

make install
