import argparse
import os

import mindspore as ms
import mindspore.nn.optim.kungfu as kungfu_mindspore_optimizer
import mindspore.ops.operations.kungfu_comm_ops as kfops
import mindspore.train.callback._kungfu as kfcallbacks
from mindspore_debug import log_duration

import debug_hook
import kungfu_elastic as ke
from dataset import create_dataset as create_dataset
from lenet import LeNet5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default='CPU',
                   choices=['Ascend', 'GPU', 'CPU'])
    p.add_argument('--data-path', type=str, default=None)
    p.add_argument('--ckpt-prefix', type=str, default='checkpoint_lenet')
    p.add_argument('--ckpt-dir', type=str, default='.')
    p.add_argument('--save-ckpt', action='store_true', default=False)
    p.add_argument('--use-kungfu', action='store_true', default=False)
    p.add_argument('--elastic', action='store_true', default=False)
    p.add_argument('--run-train', action='store_true', default=False)
    p.add_argument('--run-test', action='store_true', default=False)

    # hyperparameters
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)

    # debug options
    p.add_argument('--log-step', action='store_true', default=False)
    p.add_argument('--log-loss', action='store_true', default=False)

    return p.parse_args()


def log_args(args):
    print('device=%s' % (args.device))
    print('batch_size=%d' % (args.batch_size))
    print('epochs=%d' % (args.epochs))


def build_optimizer(args, net):
    opt = ms.nn.Momentum(
        [x for x in net.get_parameters() if x.requires_grad],
        args.learning_rate,
        args.momentum,
    )
    return opt


def build_model(args, net):
    if args.use_kungfu:
        opt = kungfu_mindspore_optimizer.build_optimizer(args, net)
    else:
        opt = build_optimizer(args, net)

    loss = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                    reduction='mean')
    model = ke.KungFuModel(
        net,
        loss_fn=loss,
        optimizer=opt,
        # loss_scale_manager=loss_scale,
        metrics={'acc'},
        amp_level="O2",
        keep_batchnorm_fp32=False,
    )
    return model


def build_ckpt_callback(prefix, directory):
    config_ck = ms.train.callback.CheckpointConfig(
        save_checkpoint_steps=100,
        keep_checkpoint_max=100,
    )
    cb = ms.train.callback.ModelCheckpoint(
        prefix=prefix,
        config=config_ck,
        directory=directory,
    )
    return cb


def get_ckpt_dir(args):
    directory = args.ckpt_dir
    if args.use_kungfu:
        rank = kfops.kungfu_current_rank()
        directory = os.path.join(directory, '%d' % (rank))
    return directory


def build_callbacks(args):
    callbacks = []
    if args.log_step:
        callbacks += [debug_hook.LogStepHook()]
    if args.log_loss:
        callbacks += [ms.train.callback.LossMonitor()]
    if args.save_ckpt:
        callbacks += [
            build_ckpt_callback(
                prefix=args.ckpt_prefix,
                directory=get_ckpt_dir(args),
            ),
        ]
    if args.elastic:
        from elastic_schedule import schedule
        callbacks += [
            kfcallbacks.KungFuElasticCallback(schedule),
        ]
    return callbacks


def train(args, model, dataset):
    # epoch_size = 1
    sink_mode = False
    model.train(
        epoch=args.epochs,
        train_dataset=dataset,
        callbacks=build_callbacks(args),
        dataset_sink_mode=sink_mode,
    )


def dump_model(net):
    for idx, p in enumerate(net.get_parameters()):
        print('[%d] %s' % (idx, p.name))


def test(net, model, ds_test, ckpt_name):
    dump_model(net)
    param_dict = ms.train.serialization.load_checkpoint(ckpt_name)
    for idx, (k, v) in enumerate(param_dict.items()):
        print('[%d] %s=%s' % (idx, k, v))
    ms.train.serialization.load_param_into_net(net, param_dict)
    acc = model.eval(ds_test, dataset_sink_mode=False)
    print("Accuracy:{}".format(acc))


def run(args):
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device,
                           save_graphs=False)

    ds_train = create_dataset(
        data_path=os.path.join(args.data_path, "train"),
        batch_size=args.batch_size,
        repeat_num=1,  #
    )

    ds_test = create_dataset(
        data_path=os.path.join(args.data_path, "test"),
        batch_size=args.batch_size,
        repeat_num=1,
    )

    step_size = ds_train.get_dataset_size()
    print('step_size: %d' % (step_size))

    net = LeNet5(num_class=10, num_channel=1)
    model = build_model(args, net)

    if args.run_train:
        train(args, model, ds_train)

    if args.run_test:
        checkpoint_steps = [(i + 1) * 100 for i in range(12)]
        checkpoints = [
            "%s-1_%d.ckpt" % (args.ckpt_prefix, n) for n in checkpoint_steps
        ]
        ckpt_dir = get_ckpt_dir(args)
        for ckpt_name in checkpoints:
            test(net, model, ds_test, os.path.join(ckpt_dir, ckpt_name))
            break


def main():
    args = parse_args()
    log_args(args)
    if args.use_kungfu:
        with kfops.KungFuContext(device=args.device):
            log_duration(run, args)
    else:
        log_duration(run, args)


# ke.info()
log_duration(main)
