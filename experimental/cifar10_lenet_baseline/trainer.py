import os
import time

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
from mindspore_kungfu_debug import (LogStepHook, load_ckpt, get_ckpt_dir,
                                    KungFuModel)
from mindspore.train.loss_scale_manager import FixedLossScaleManager


def build_ckpt_callback(prefix, directory):
    config_ck = ms.train.callback.CheckpointConfig(
        save_checkpoint_steps=10,
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
    loss_scale = FixedLossScaleManager(
        loss_scale=1,
        drop_overflow_update=False,
    )
    # model = KungFuModel(
    model = ms.train.model.Model(
        net,
        loss_fn=loss,
        optimizer=opt,
        loss_scale_manager=loss_scale,
        metrics={'acc'},
        amp_level="O2",
        keep_batchnorm_fp32=False,
    )
    model.train(
        # args,
        epoch=args.epochs,
        train_dataset=dataset,
        callbacks=build_callbacks(args),
        dataset_sink_mode=False,
    )


def test_checkpoint(net, model, ds_test, ckpt_name):
    print('testing %s' % (ckpt_name))
    # dump_model(net)
    load_ckpt(net, ckpt_name)
    net.set_train(False)
    print('loaded %s' % (ckpt_name))
    result = model.eval(
        ds_test,
        dataset_sink_mode=False,
    )
    return result


def parse_logical_step(ckpt_name):
    # ckpt_name = 'x/x/x/mnist-slp-000000.npz'
    return int(ckpt_name.split('/')[-1].split('.')[-2].split('-')[-1])


def get_eval_result_filename(args):
    timestamp = time.time()
    filename = 'lbs-%d+dbs-%d-%d.txt' % (args.logical_batch_size,
                                         args.device_batch_size, timestamp)
    if args.use_kungfu:
        rank = kfops.kungfu_current_rank()
        filename = 'worker.%d.%s' % (rank, filename)
    filename = os.path.join('plot', filename)
    return filename


def test(args, net, loss, opt, dataset, checkpoints):
    metrics = [
        'top_1_accuracy',
        'top_5_accuracy',
    ]
    model = KungFuModel(
        net,
        loss_fn=loss,
        optimizer=opt,
        metrics=set(metrics),
        amp_level="O2",

        # [ERROR] DEVICE(8546,python3.7):2021-02-27-01:12:19.225.728 [mindspore/ccsrc/runtime/device/gpu/kernel_info_setter.cc:118] SelectAkgKernel] Not find op[BatchNorm] in akg
        # [ERROR] DEVICE(8546,python3.7):2021-02-27-01:12:19.225.791 [mindspore/ccsrc/runtime/device/gpu/kernel_info_setter.cc:322] PrintUnsupportedTypeException] Select GPU kernel op[BatchNorm] fail! Incompatible data type!
        # The supported data types are in[float32 float32 float32 float32 float32], out[float32 float32 float32 float32 float32]; in[float16 float32 float32 float32 float32], out[float16 float32 float32 float32 float32]; , but get in [float16 float16 float16 float16 float16 ] out [float16 float16 float16 float16 float16 ]
        # keep_batchnorm_fp32=False,
    )

    results = []
    for ckpt_name in checkpoints:
        logical_step = parse_logical_step(ckpt_name)
        result = test_checkpoint(net, model, dataset, ckpt_name)
        results.append((logical_step, result))
        msg = "%s, %s: %s" % (ckpt_name, metrics[0], result[metrics[0]])
        print(msg)

    filename = get_eval_result_filename(args)

    with open(filename, 'w') as f:
        for step, result in results:
            acc = result[metrics[0]]
            msg = '%d %s' % (step, acc)
            f.write(msg + '\n')

    print('%d points saved to %s' % (len(results), filename))
