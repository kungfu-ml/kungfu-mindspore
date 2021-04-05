import argparse
import os
import sys
import numpy as np
import mindspore.dataset as ds
import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
import mindspore.dataset.engine as de

from cifar10 import create_dataset1 as create_dataset
from elastic_dataset import create_elastic_mnist
# from elastic_state import State


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


def elastic_dataset_example(args):
    data_dir = os.path.join(args.data_path, 'mnist', 'train')
    dataset = create_elastic_mnist(
        data_path=data_dir,
        batch_size=args.batch_size,
    )
    # <mindspore.dataset.engine.datasets.BatchDataset object at 0x7fc989f84350>
    print('[Python] create_elastic_mnist returned %s' % (dataset),
          file=sys.stderr)
    total = dataset.get_dataset_size()
    print(
        'total steps: %d when using batch size: %d' % (total, args.batch_size),
        file=sys.stderr,
    )

    it = enumerate(dataset)
    print('enumerate(dataset) returned %s' % (it), file=sys.stderr)
    for i in range(min(args.max_step, total)):
        print('iteration step %d' % (i), file=sys.stderr)
        idx, (x, y) = next(it)
        print(
            'data consumed: %d/%d %s%s %s%s' %
            (idx, total, x.dtype, x.shape, y.dtype, y.shape),
            file=sys.stderr,
        )


def main(args):
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target=args.device,
        save_graphs=False,
    )

    elastic_dataset_example(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
