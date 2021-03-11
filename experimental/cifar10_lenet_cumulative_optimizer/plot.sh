#!/bin/sh
set -e

cd $(dirname $0)

# pdflatex plot.tex
pdflatex plot-moment-without-bn.tex
pdflatex plot-moment-with-bn.tex
