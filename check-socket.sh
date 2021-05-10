#!/bin/sh
set -e

for s in $(ls /tmp/kungfu-run-*.sock); do
    echo $s
    sudo rm -fr $s
done
