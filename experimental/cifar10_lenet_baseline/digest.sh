#!/bin/sh
set -e

d() {
    cd $1
    md5sum *
    cd -
    echo
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

digest_all >digest.txt
