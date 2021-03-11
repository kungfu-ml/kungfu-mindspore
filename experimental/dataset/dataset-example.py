import argparse

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np

from cifar10 import create_dataset1 as create_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default='CPU',
                   choices=['Ascend', 'GPU', 'CPU'])
    p.add_argument('--dataset_path', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=100)
    return p.parse_args()


def log_args(args):
    print('device=%s' % (args.device))


def train(args):
    with kfops.KungFuContext(device=args.device):
        all_reduce = kfops.KungFuAllReduce()
        x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
        print(x)
        y = all_reduce(x)
        print(y)


def main():
    args = parse_args()
    log_args(args)
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device,
                           save_graphs=False)

    dataset = create_dataset(dataset_path=args.dataset_path,
                             do_train=True,
                             repeat_num=1,
                             batch_size=args.batch_size,
                             target=args.device)
    step_size = dataset.get_dataset_size()

    print('step_size: %d' % (step_size))
    train(args)


main()
