import mindspore as ms
import os
import numpy as np

from mindspore_kungfu_debug import LogStepHook
from mindspore.train.serialization import save_checkpoint
import mindspore.ops.operations.kungfu_comm_ops as kfops


def get_ckpt_dir(args):
    directory = args.ckpt_dir
    if args.use_kungfu:
        rank = kfops.kungfu_current_rank()
        directory = os.path.join(directory, '%d' % (rank))
    return directory


def get_ckpt_file_name(args, step, suffix='ckpt'):
    filename = '%s-%06d.%s' % (args.ckpt_prefix, step, suffix)
    return os.path.join(args.ckpt_dir, filename)


def get_ckpt_file_name_2(args, device_step, suffix='ckpt'):
    filename = '%s-device-%06d+%d.%s' % (
        args.ckpt_prefix,
        device_step,
        args.device_batch_size,
        suffix,
    )
    return os.path.join(args.ckpt_dir, filename)


def save_npz(net, filename):
    values = dict()
    for p in net.get_parameters():
        name = p.name
        v = p.asnumpy()
        values[name] = v
        # print('name=%s' % (name))
    np.savez(filename, **values)


def save_data_npz(x, y_, filename):
    np.savez(filename, samples=x.asnumpy(), labels=y_.asnumpy())


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


class KungFuModel(ms.train.model.Model):  # replace ms.train.model.Model
    def __init__(self, *args, **kwargs):
        print('%s::%s' % ('KungFuModel', '__init__'))
        super(KungFuModel, self).__init__(*args, **kwargs)
        # TODO: auto inject kungfu callback

    def train(self, *args, **kwargs):
        print('%s::%s' % ('KungFuModel', 'train'))
        # TODO: auto inject kungfu callback
        super(KungFuModel, self).train(*args, **kwargs)


def train_old(args, net, loss, opt, dataset):
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
        epoch=args.epochs,
        train_dataset=dataset,
        callbacks=build_callbacks(args),
        dataset_sink_mode=False,
    )


def train_new(args, net, loss, opt, dataset):
    step_count = dataset.get_dataset_size()
    print('step_count: %d' % (step_count))

    apply_period = args.logical_batch_size / args.device_batch_size

    save_period = 1
    while step_count / save_period >= 20:
        save_period += 1
    print('save_period: %d' % (save_period))

    train_net = ms.train.amp.build_train_network(
        net,
        opt,
        loss,
        level="O0",
        keep_batchnorm_fp32=False,
    )

    dataset_helper = ms.train.dataset_helper.DatasetHelper(
        dataset=dataset,
        dataset_sink_mode=False,
        sink_size=1,
        epoch_num=args.epochs,
    )

    save_npz(net, get_ckpt_file_name(args, 0, 'npz'))
    save_npz(net, get_ckpt_file_name_2(args, 0, 'npz'))

    device_step = 0
    for i in range(args.epochs):
        print('epoch #%d started' % (i + 1))
        for batch in dataset_helper:
            device_step += 1
            print('device step %d' % (device_step))

            # x, y_ = batch
            # save_data_npz(x, y_, 'batch-%05d.npz' % (device_step))

            train_net(*batch)

            save_npz(net, get_ckpt_file_name_2(args, device_step, 'npz'))

            if device_step % apply_period == 0:
                logical_step = device_step / apply_period
                print('physical step %d, logical_step %d' %
                      (device_step, logical_step))
                if logical_step % args.ckpt_period == 0:  # or logical_step < 10:
                    save_checkpoint(
                        net,
                        ckpt_file_name=get_ckpt_file_name(args, logical_step),
                    )
                    save_npz(net, get_ckpt_file_name(args, logical_step,
                                                     'npz'))

                if args.stop_logical_step > 0 and logical_step == args.stop_logical_step:
                    print('early stop at logical_step %d' % (logical_step))
                    return

        dataset.reset()
        print('epoch #%d finished' % (i + 1))


def train(args, net, loss, opt, dataset):
    print('BEGIN :: train')
    # train_old(args, net, loss, opt, dataset)
    train_new(args, net, loss, opt, dataset)
    print('END :: train')


def dump_model(net):
    for idx, p in enumerate(net.get_parameters()):
        print('[%d] %s' % (idx, p.name))


def load_ckpt(net, ckpt_name):
    param_dict = ms.train.serialization.load_checkpoint(ckpt_name)
    # for idx, (k, v) in enumerate(param_dict.items()):
    #     print('[%d] %s=%s' % (idx, k, v))
    ms.train.serialization.load_param_into_net(net, param_dict)


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
