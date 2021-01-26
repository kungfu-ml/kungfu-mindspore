#!/bin/sh
set -e

cd $(dirname $0)
. ../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../mindspore)

np=4
# np=2

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -logfile kungfu-run.log
    echo -np $np
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

app_flags() {
    # echo --device CPU
    echo --device GPU

    echo --warmup-steps 2
    echo --steps 10

    # echo --model empty
    # echo --model vgg16
    echo --model resnet50
}

mpi_run_flags() {
    echo -np $np
}

mpi_run() {
    mpirun $(mpi_run_flags) $@
}

trace() {
    echo "BEGIN $@"
    $@
    echo "END $@"
    echo
    echo
}

main() {
    rm -fr logs
    # trace kungfu_run python3.7 ./benchmark_all_reduce.py $(app_flags)
    # trace mpi_run python3.7 ./benchmark_all_reduce.py $(app_flags)
    trace kungfu_run python3.7 ./hello_world.py --device GPU
}

main
