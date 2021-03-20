import argparse
import os
import numpy as np
import mindspore.dataset as ds
import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
import mindspore.dataset.engine as de

from cifar10 import create_dataset1 as create_dataset
from elastic_dataset import create_elastic_mnist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default='CPU',
                   choices=['Ascend', 'GPU', 'CPU'])
    p.add_argument('--data-path', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--max-step', type=int, default=100)
    return p.parse_args()


def log_args(args):
    print('device=%s' % (args.device))


def elastic_example(args):
    data_dir = os.path.join(args.data_path, 'mnist', 'train')
    dataset = create_elastic_mnist(
        data_path=data_dir,
        batch_size=args.batch_size,
    )
    total = dataset.get_dataset_size()
    print('total steps: %d when using batch size: %d' % (
        total,
        args.batch_size,
    ))

    total_step = total / args.batch_size

    with kfops.KungFuContext(device=args.device):
        it = enumerate(dataset)

        for i in range(min(args.max_step, total_step)):
            idx, (x, y) = next(it)
            print('%d/%d %s%s %s%s' %
                  (idx, total, x.dtype, x.shape, y.dtype, y.shape))

        # all_reduce = kfops.KungFuAllReduce()
        # x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
        # print(x)
        # y = all_reduce(x)
        # print(y)
        # TODO: use elastic

    # print('step_size: %d' % (total))
    # for idx, (x, y) in enumerate(dataset):
    #     print('%d/%d %s%s %s%s' %
    #           (idx, total, x.dtype, x.shape, y.dtype, y.shape))
    #     # print(x.__class__)  # <class 'mindspore.common.tensor.Tensor'>


def main(args):
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target=args.device,
        save_graphs=False,
    )

    elastic_example(args)


if __name__ == '__main__':
    args = parse_args()
    log_args(args)
    main(args)
