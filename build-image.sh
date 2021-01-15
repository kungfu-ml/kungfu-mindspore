#!/bin/sh
set -e

cd $(dirname $0)
. ./scripts/measure.sh

TAG=mindspore-kungfu:snapshot
measure docker build -t $TAG .

# docker run --rm -it $TAG
