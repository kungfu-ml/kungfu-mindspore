#!/bin/sh
set -e

cd $(dirname $0)

pdflatex plot-sgd-without-bn.tex
pdflatex plot-sgd-with-bn.tex
pdflatex plot-moment-without-bn.tex
pdflatex plot-moment-with-bn.tex

pdflatex plot.tex

rm plot*.log
rm *.aux
rm *.nav
rm *.out
rm *.snm
rm *.toc
