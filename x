#!/bin/sh
set -e

cd $(dirname $0)
. ./debug_options.sh

# ./debug/run.sh
# ./benchmarks/run.sh

# ./examples/resnet-elastic/train_single.sh
./examples/resnet-elastic/train_parallel_kungfu.sh
# ./examples/resnet-elastic/train_parallel_kungfu_elastic.sh
# ./examples/resnet-elastic/train_parallel_mpi.sh

# ./experimental/elastic/run.sh
