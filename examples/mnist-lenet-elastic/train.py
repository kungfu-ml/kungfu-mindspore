import argparse
import os

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
from download import download_dataset
from kungfu_mindspore_optimizer import build_optimizer
from mindspore._c_expression import kungfu_finalize, kungfu_init
from mindspore.common.initializer import Normal
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint)
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from dataset import create_dataset
from model import LeNet5


def parse_args():
    p = argparse.ArgumentParser(description='MindSpore LeNet Example')
    p.add_argument(
        '--device',
        type=str,
        default="CPU",
        choices=['Ascend', 'GPU', 'CPU'],
        help='device where the code will be implemented (default: CPU)')
    p.add_argument('--data-dir', type=str, default="MNIST_Data")
    p.add_argument('--epoch-size', type=int, default=1)
    p.add_argument('--repeat-size', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--use-kungfu', action='store_true', default=False)
    p.add_argument('--use-kungfu-elastic', action='store_true', default=False)
    p.add_argument('--run-test', action='store_true', default=False)

    p.add_argument('--init-ckpt', type=str, default='')
    return p.parse_args()


def log_callbacks(cb):
    print('%d callbacks' % (len(cb)))
    for c in cb:
        print('%s' % (c))


def load_ckpt(net, ckpt_name):
    param_dict = ms.train.serialization.load_checkpoint(ckpt_name)
    ms.train.serialization.load_param_into_net(net, param_dict)


def train_net(network, model, args, ckpoint_cb, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    ds_train = create_dataset(os.path.join(args.data_dir, "train"),
                              args.batch_size, args.repeat_size)

    callbacks = [
        # ckpoint_cb,
        LossMonitor(per_print_times=20),
    ]

    if args.use_kungfu:
        if args.use_kungfu_elastic:
            from kungfu_mindspore_callbacks import KungFuElasticCallback
            schedule = {
                10: 2,
                20: 3,
                30: 4,
                40: 1,
                50: 2,
                60: 3,
                70: 4,
                80: 1,
            }
            kungfu_elastic_callback = KungFuElasticCallback(schedule)
            callbacks.append(kungfu_elastic_callback)

    log_callbacks(callbacks)
    print('sink_mode: %s' % (sink_mode))
    model.train(args.epoch_size,
                ds_train,
                callbacks=callbacks,
                dataset_sink_mode=sink_mode)


def test_net(network, model, args):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
    # load parameter to the network
    load_param_into_net(network, param_dict)
    # load testing dataset
    ds_eval = create_dataset(os.path.join(args.data_dir, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


def run(args):
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)
    dataset_sink_mode = False

    download_dataset(args.data_dir)

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # create the network
    network = LeNet5()
    # define the optimizer
    net_opt = build_optimizer(args, network)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875,
                                 keep_checkpoint_max=10)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    # group layers into an object with training and evaluation features
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    if args.init_ckpt:
        load_ckpt(network, args.init_ckpt)

    train_net(network, model, args, ckpoint_cb, dataset_sink_mode)
    # if args.run_test:
    #     test_net(network, model, args.data_dir)


def main():
    args = parse_args()
    with kfops.KungFuContext(device=args.device):
        run(args)


main()
