#!/bin/sh
join_path() {
    local IFS=":"
    echo "$*"
}

ld_library_paths() {
    local MS_ROOT=$1
    local KUNGFU_LIB_PATH=$MS_ROOT/third_party/kungfu/lib
    echo $KUNGFU_LIB_PATH
    echo $MS_ROOT/mindspore/lib
    echo $MS_ROOT/build/mindspore/_deps/ompi-src/ompi/.libs
    echo $MS_ROOT/build/mindspore/_deps/nccl-src/build/lib
}

ld_library_path() {
    local MS_ROOT=$1
    join_path $(ld_library_paths $MS_ROOT)
}
