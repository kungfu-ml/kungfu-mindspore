#!/bin/bash
set -e

cd $(dirname $0)

# pdflatex plot.tex
pdflatex plot-moment-without-bn.tex
pdflatex plot-moment-with-bn.tex

plot_dir() {
    pushd $1
    pdflatex plot.tex
    popd
}

plot_dir plot/with-bn-logical-bs
plot_dir plot/with-bn2-logical-bs
plot_dir plot/without-bn-logical-bs
