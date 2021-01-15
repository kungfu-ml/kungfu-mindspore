#!/bin/sh
set -e

cd $(dirname $0)
. ../scripts/measure.sh

TAG=mindspore-builder:1.1.0
measure docker build -t $TAG -f Dockerfile.builder .
