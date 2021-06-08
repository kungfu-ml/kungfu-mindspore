#!/bin/sh
set -e

cd $(dirname $0)/mindspore

reinstall() {
    cd output
    whl=$(ls *.whl)
    echo $whl
    python -m pip install -U ./$whl
}

reinstall
