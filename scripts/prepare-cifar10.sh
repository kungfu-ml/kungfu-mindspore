#!/bin/sh
set -e

mkdir -p $HOME/var/data/mindspore
cd $HOME/var/data/mindspore

mkdir -p cifar10
cd cifar10

mkdir -p train
mkdir -p test

cp $HOME/var/data/cifar/cifar-10-batches-bin/data_batch_*.bin train
cp $HOME/var/data/cifar/cifar-10-batches-bin/test_batch.bin test

tree
