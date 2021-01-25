#!/bin/sh
set -e

/usr/local/cuda/bin/cuobjdump ./mindspore/build/mindspore/_deps/nccl-src/build/lib/libnccl.so
