import argparse
import os
import time

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
from mindspore_kungfu_debug import CumulativeSGDOptimizer, log_duration, CumulativeMomentumOptimizer
from mindspore_kungfu_debug.cifar10_slp import Cifar10SLP
from dataset import create_dataset as create_dataset
from trainer import train, test, get_ckpt_file_name, load_ckpt
from mindspore.train.serialization import save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()

    # hardware parameters
    p.add_argument('--device',
                   type=str,
                   default='CPU',
                   choices=['Ascend', 'GPU', 'CPU'])
    p.add_argument('--device-batch-size', type=int, default=100)

    # config parameters
    p.add_argument('--data-path', type=str, default=None)
    p.add_argument('--ckpt-prefix', type=str, default='cifar10-slp')
    p.add_argument('--ckpt-dir', type=str, default='.')
    p.add_argument('--save-ckpt', action='store_true', default=False)
    p.add_argument('--use-kungfu', action='store_true', default=False)
    p.add_argument('--elastic', action='store_true', default=False)
    p.add_argument('--ckpt-files', type=str, default='')
    p.add_argument('--ckpt-period', type=int, default=1)
    p.add_argument('--init-ckpt', type=str, default='')
    p.add_argument('--mode',
                   type=str,
                   default='train',
                   choices=['train', 'test', 'init'])

    # hyper parameters
    p.add_argument('--logical-batch-size', type=int, default=100)
    p.add_argument('--test-batch-size', type=int, default=100)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--optimizer',
                   type=str,
                   default='momentum',
                   choices=['momentum', 'sgd'])
    p.add_argument('--learning-rate', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)

    # debug options
    p.add_argument('--log-step', action='store_true', default=False)
    p.add_argument('--log-loss', action='store_true', default=False)
    p.add_argument('--stop-logical-step', type=int, default=0)

    return p.parse_args()


def log_args(args):
    print('--- log args BEGIN ---------------')
    # hardware parameters
    print('device=%s' % (args.device))
    print('device_batch_size=%d' % (args.device_batch_size))

    # hyper parameters
    print('epochs=%d' % (args.epochs))
    print('logical_batch_size=%d' % (args.logical_batch_size))
    print('learning rage=%f' % (args.learning_rate))
    print('momentum=%f' % (args.momentum))

    print('--- log args END -----------------')


def build_optimizer(args, net):
    # if args.use_kungfu:
    #     return kungfu_mindspore_optimizer.build_optimizer(args, net)

    if args.logical_batch_size % args.device_batch_size != 0:
        msg = '--logical-batch-size (%d) is not a multiple of --device-batch-size (%s)' % (
            args.logical_batch_size, args.device_batch_size)
        raise RuntimeError(msg)

    apply_period = args.logical_batch_size / args.device_batch_size

    bypass = False

    if args.optimizer == 'sgd':
        if bypass and apply_period == 1:
            print('using SGD')
            opt = ms.nn.SGD(
                [x for x in net.get_parameters() if x.requires_grad],
                args.learning_rate,
            )
        else:
            print('using CumulativeSGDOptimizer with apply_period=%d' %
                  (apply_period))
            opt = CumulativeSGDOptimizer(
                [x for x in net.get_parameters() if x.requires_grad],
                args.learning_rate,
                apply_period=apply_period,
            )
    elif args.optimizer == 'momentum':
        if bypass and apply_period == 1:
            print('using Momentum')
            opt = ms.nn.Momentum(
                [x for x in net.get_parameters() if x.requires_grad],
                args.learning_rate,
                args.momentum,
            )
        else:
            print('using CumulativeMomentumOptimizer with apply_period=%d' %
                  (apply_period))
            opt = CumulativeMomentumOptimizer(
                [x for x in net.get_parameters() if x.requires_grad],
                args.learning_rate,
                args.momentum,
                apply_period=apply_period,
            )
    else:
        msg = 'invalid optimizer: %s' % (args.optimizer)
        raise RuntimeError(msg)
    return opt


def run(args):
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target=args.device,
        save_graphs=False,
    )

    net = Cifar10SLP(num_class=10, num_channel=1)
    loss = ms.nn.loss.SoftmaxCrossEntropyWithLogits(
        sparse=True,
        reduction='mean',
    )
    opt = build_optimizer(args, net)

    if args.mode == 'init':
        save_checkpoint(
            net,
            ckpt_file_name=os.path.join('seeds',
                                        'cifar10-slp-%d.ckpt' % (time.time())),
        )

    if args.mode == 'train':
        ds_train = create_dataset(
            data_path=os.path.join(args.data_path, "train"),
            batch_size=args.device_batch_size,
        )

        if args.init_ckpt:
            print('using init checkpoint %s' % (args.init_ckpt))
            load_ckpt(net, args.init_ckpt)
        train(args, net, loss, opt, ds_train)

    if args.mode == 'test':
        ds_test = create_dataset(
            data_path=os.path.join(args.data_path, "test"),
            batch_size=args.test_batch_size,
        )

        if args.ckpt_files:
            checkpoints = args.ckpt_files.split(',')
        else:
            steps = [10, 20, 30, 40]
            checkpoints = [get_ckpt_file_name(args, i) for i in steps]
        print('will test %d checkpoints' % (len(checkpoints)))
        # for i, n in enumerate(checkpoints):
        #     print('[%d]=%s' % (i, n))
        test(args, net, loss, opt, ds_test, checkpoints)


def main():
    args = parse_args()
    log_args(args)
    if args.use_kungfu:
        with kfops.KungFuContext(device=args.device):
            log_duration(run, args)
    else:
        # log_duration(run, args)
        run(args)

    log_args(args)


# ke.info()
# log_duration(main)
main()
