import os

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
from mindspore_kungfu_debug import (LogStepHook, load_ckpt, KungFuModel)


def get_ckpt_dir(args):
    directory = args.ckpt_dir
    if args.use_kungfu:
        rank = kfops.kungfu_current_rank()
        directory = os.path.join(directory, '%d' % (rank))
    return directory


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


def build_callbacks(args):
    callbacks = []
    if args.log_step:
        callbacks += [LogStepHook()]
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


def train(args, net, loss, opt, dataset):
    print('BEGIN :: train')
    model = KungFuModel(
        net,
        loss_fn=loss,
        optimizer=opt,
        # loss_scale_manager=loss_scale,
        metrics={'acc'},
        amp_level="O2",
        keep_batchnorm_fp32=False,
    )
    model.train(
        args,
        epoch=args.epochs,
        train_dataset=dataset,
        callbacks=build_callbacks(args),
        dataset_sink_mode=False,
    )
    print('END :: train')


def test_checkpoint(net, model, ds_test, ckpt_name):
    # dump_model(net)
    load_ckpt(net, ckpt_name)
    result = model.eval(ds_test, dataset_sink_mode=False)
    return result['acc']


def parse_logical_step(ckpt_name):
    # ckpt_name = 'x/x/x/mnist-slp-000000.npz'
    return int(ckpt_name.split('/')[-1].split('.')[-2].split('-')[-1])


def test(args, net, loss, opt, dataset, checkpoints):
    model = KungFuModel(
        net,
        loss_fn=loss,
        optimizer=opt,
        # loss_scale_manager=loss_scale,
        metrics={'acc'},
        amp_level="O2",
        keep_batchnorm_fp32=False,
    )

    results = []
    for ckpt_name in checkpoints:
        logical_step = parse_logical_step(ckpt_name)
        acc = test_checkpoint(net, model, dataset, ckpt_name)
        results.append((logical_step, acc))
        msg = "%s, accuracy: %s" % (ckpt_name, acc)
        print(msg)

    filename = 'plot/lbs-%d+dbs-%d.txt' % (args.logical_batch_size,
                                           args.device_batch_size)

    with open(filename, 'w') as f:
        for step, acc in results:
            # msg = "%s, accuracy: %s" % (ckpt_name, acc)
            msg = '%d %s' % (step, acc)
            f.write(msg + '\n')

    print('saved to %s' % (filename))
