#!/bin/sh
set -e
cd $(dirname $0)

TAG=mindspore-builder:1.1.0
docker build -t $TAG -f Dockerfile.builder .
