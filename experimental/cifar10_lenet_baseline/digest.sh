#!/bin/sh
set -e

cd $(dirname $0)
W=$PWD

read_npz=$W/../../scripts/read-npz

d() {
    cd $1
    md5sum * >md5.txt
    $read_npz *.npz >digest.txt
    cd -
    # echo
    # echo
    # echo
}

digest_all() {
    d checkpoint/1-0

    d checkpoint/2-0
    d checkpoint/2-1

    d checkpoint/3-0
    d checkpoint/3-1
    d checkpoint/3-2

    d checkpoint/4-0
    d checkpoint/4-1
    d checkpoint/4-2
    d checkpoint/4-3
}

digest_all
# >digest.txt
