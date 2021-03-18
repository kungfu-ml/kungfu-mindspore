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


def numpy_ds_example():
    np.random.seed(6)
    features, labels = np.random.sample((4, 2)), np.random.sample((4, 1))

    data = (features, labels)
    dataset = ds.NumpySlicesDataset(data,
                                    column_names=["col1", "col2"],
                                    shuffle=False)

    for data in dataset:
        print(data[0], data[1])


def cifar10_example():
    data_dir = os.path.join(args.data_path, 'cifar10', 'train')
    print(data_dir)
    dataset = create_dataset(
        dataset_path=data_dir,
        do_train=True,
        repeat_num=1,
        batch_size=args.batch_size,
        target=args.device,
    )
    step_size = dataset.get_dataset_size()
    print('step_size: %d' % (step_size))


def elastic_example():
    data_dir = os.path.join(args.data_path, 'mnist', 'train')
    dataset = create_elastic_mnist(
        data_path=data_dir,
        batch_size=1000,
    )
    total = dataset.get_dataset_size()

    print('step_size: %d' % (total))
    for idx, (x, y) in enumerate(dataset):
        print('%d/%d %s%s %s%s' %
              (idx, total, x.dtype, x.shape, y.dtype, y.shape))
        # print(x.__class__)  # <class 'mindspore.common.tensor.Tensor'>


def main(args):
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target=args.device,
        save_graphs=False,
    )

    numpy_ds_example()
    cifar10_example()
    elastic_example()

    train(args)


if __name__ == '__main__':
    args = parse_args()
    log_args(args)
    main(args)
