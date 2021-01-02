#!/bin/sh
set -e

if [ ! -d mindspore ]; then
    git clone https://gitee.com/mindspore/mindspore.git
fi
cd mindspore
git checkout $(cat ../tag.txt)

./build.sh -e gpu
