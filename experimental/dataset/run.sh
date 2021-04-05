#!/bin/sh

cd $(dirname $0)
. ../../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path ../../mindspore)
. ../../scripts/launcher.sh

app_flags() {
    echo --device GPU
    echo --data-path $HOME/var/data/mindspore

    echo --batch-size 8
    # echo --batch-size 1000

    echo --max-step 10
}

main() {
    rm -fr logs

    local np=1
    # prun $np ./dataset-example.py $(app_flags)

    # srun ./dataset-example.py $(app_flags)

    # erun 1 ./dataset-example.py $(app_flags)
    # erun 1 ./elastic-main.py $(app_flags)
    # erun 2 ./elastic-main.py $(app_flags)

    srun ./simple-main.py $(app_flags)
}

main
