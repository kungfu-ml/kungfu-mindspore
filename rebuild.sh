#!/bin/sh
set -e

cd $(dirname $0)
. ./scripts/measure.sh

rebuild() {
    rm -fr mindspore
    ./prebuild.sh
    ./build.sh
    ./install.sh
}

measure rebuild
