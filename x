#!/bin/sh
set -e

# export NCCL_DEBUG=INFO
# export KUNGFU_MINDSPORE_DEBUG=true
# export KUNGFU_USE_NCCL_SCHEDULER=true

# export GLOG_v=3 # ERROR
# export GLOG_v=2 # WARNING
# export GLOG_v=1 # INFO
# export GLOG_v=0 # DEBUG

# ./benchmarks/run.sh
# ./examples/resnet-elastic/train_single.sh
# ./examples/resnet-elastic/train_parallel_kungfu.sh
./examples/resnet-elastic/train_parallel_kungfu_elastic.sh
# ./examples/resnet-elastic/train_parallel_mpi.sh

# ./experimental/elastic/run.sh
# ./debug/run.sh
