import argparse

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default='CPU',
                   choices=['Ascend', 'GPU', 'CPU'])
    return p.parse_args()


def log_args(args):
    print('device=%s' % (args.device))


def main():
    args = parse_args()
    log_args(args)
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device,
                           save_graphs=False)

    kfops.init(args.device)

    all_reduce = kfops.KungFuAllReduce()

    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    print(x)
    y = all_reduce(x)
    print(y)

    kfops.finalize(args.device)


main()
