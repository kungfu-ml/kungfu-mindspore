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


class KungFuContext:
    def __init__(self, device='CPU'):
        self._device = device

    def __enter__(self):
        kfops.kungfu_init()

    def __exit__(self, exc_type, exc_val, exc_tb):
        kfops.kungfu_finalize()


def main():
    args = parse_args()
    log_args(args)
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device,
                           save_graphs=False)

    with KungFuContext():
        kfops.kungfu_debug_nccl()


main()
