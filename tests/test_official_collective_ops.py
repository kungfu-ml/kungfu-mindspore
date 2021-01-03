import argparse

import mindspore as ms
import numpy as np
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.ops.operations.comm_ops import AllReduce, Broadcast

dtype_map = {
    'i32': np.int32,
    'f32': np.float32,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='CPU', choices=['GPU', 'CPU'])
    p.add_argument('--dtype', type=str, default='f32', choices=['i32', 'f32'])
    p.add_argument('--op',
                   type=str,
                   default='all_reduce',
                   choices=['all_reduce', 'broadcast'])
    return p.parse_args()


def test_all_reduce(x):
    print('test_all_reduce with %s' % (x))
    all_reduce = AllReduce()
    y = all_reduce(x)
    print('y=%s' % (y))


def test_broadcast(x):
    print('test_broadcast with %s' % (x))
    broadcast = Broadcast(0)
    y = broadcast(x)
    print('y=%s' % (y))


op_map = {
    'all_reduce': test_all_reduce,
    'broadcast': test_broadcast,
}


def main():
    args = parse_args()
    dtype = dtype_map[args.dtype]
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    init()
    rank = get_rank()
    size = get_group_size()
    print('%d/%d' % (rank, size))

    test_fn = op_map[args.op]

    value = 1
    n = 10
    x = ms.Tensor(np.array([value] * n).astype(dtype))
    test_fn(x)


main()
