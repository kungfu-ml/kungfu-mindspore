#!/usr/bin/env python3
import mindspore as ms
import sys


def main(filename):
    param_dict = ms.train.serialization.load_checkpoint(filename)
    for idx, (k, p) in enumerate(param_dict.items()):
        x = p.asnumpy()
        print('[%3d] %-24s: %s%-20s [%f, %f] ~ %f' % (
            idx,
            k,
            x.dtype,
            x.shape,
            x.min(),
            x.max(),
            x.mean(),
        ))


main(sys.argv[1])
