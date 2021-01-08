#!/bin/sh
set -e

# export GLOG_v=3 # ERROR
# export GLOG_v=2 # WARNING
# export GLOG_v=1 # INFO
# export GLOG_v=0 # DEBUG

# ./examples/resnet-elastic/train_single.sh
./examples/resnet-elastic/train_parallel_kungfu.sh
# ./examples/resnet-elastic/train_parallel_kungfu_elastic.sh
# ./examples/resnet-elastic/train_parallel_mpi.sh
