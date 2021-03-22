#!/bin/sh
set -e

cd $(dirname $0)

pdflatex plot-all.tex
