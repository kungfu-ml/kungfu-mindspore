import os

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
from mindspore.train.serialization import save_checkpoint
from mindspore_kungfu_debug import LogStepHook

from .utils import save_npz, save_npz_per_weight


def get_ckpt_dir(args):
    directory = args.ckpt_dir
    if args.use_kungfu:
        rank = kfops.kungfu_current_rank()
        directory = os.path.join(directory, '%d' % (rank))
    return directory


def get_ckpt_file_name(args, step, suffix='ckpt'):
    filename = '%s-%06d.%s' % (args.ckpt_prefix, step, suffix)
    return os.path.join(get_ckpt_dir(args), filename)


def get_ckpt_file_name_2(args, device_step, suffix='ckpt'):
    filename = '%s-device-%06d+%d.%s' % (
        args.ckpt_prefix,
        device_step,
        args.device_batch_size,
        suffix,
    )
    return os.path.join(get_ckpt_dir(args), filename)


def get_ckpt_file_name_3(args, step, name):
    filename = '%s-%06d-%s.%s' % (args.ckpt_prefix, step, name, 'npz')
    return os.path.join(args.ckpt_dir, filename)


class KungFuModel(ms.train.model.Model):  # replace ms.train.model.Model
    def __init__(self, *args, **kwargs):
        print('%s::%s' % ('KungFuModel', '__init__'))
        super(KungFuModel, self).__init__(*args, **kwargs)
        # TODO: auto inject kungfu callback

    def train(
        self,
        args,
        epoch,
        train_dataset,
        callbacks,  #  FIXME: not used now
        dataset_sink_mode,
    ):
        net = self._network
        step_count = train_dataset.get_dataset_size()
        print('step_count: %d' % (step_count))

        apply_period = args.logical_batch_size / args.device_batch_size

        train_net = self._train_network

        dataset_helper = ms.train.dataset_helper.DatasetHelper(
            dataset=train_dataset,
            dataset_sink_mode=dataset_sink_mode,
            sink_size=1,
            epoch_num=epoch,
        )

        save_npz(net, get_ckpt_file_name(args, 0, 'npz'))
        save_npz(net, get_ckpt_file_name_2(args, 0, 'npz'))
        # save_npz_per_weight(net,
        #                     lambda name: get_ckpt_file_name_3(args, 0, name))

        device_step = 0
        for i in range(args.epochs):
            print('epoch #%d started' % (i + 1))
            for batch in dataset_helper:
                device_step += 1
                print('device step %d' % (device_step))

                # x, y_ = batch
                # save_data_npz(x, y_, 'batch-%05d.npz' % (device_step))

                train_net(*batch)

                # save_npz(net, get_ckpt_file_name_2(args, device_step, 'npz'))

                if device_step % apply_period == 0:
                    logical_step = device_step / apply_period
                    print('physical step %d, logical_step %d' %
                          (device_step, logical_step))
                    if logical_step % args.ckpt_period == 0:  # or logical_step < 10:
                        # FIXME: call hooks
                        save_checkpoint(
                            net,
                            ckpt_file_name=get_ckpt_file_name(
                                args, logical_step),
                        )
                        save_npz(net,
                                 get_ckpt_file_name(args, logical_step, 'npz'))
                        # save_npz_per_weight(
                        #     net, lambda name: get_ckpt_file_name_3(
                        #         args, logical_step, name))

                    if args.stop_logical_step > 0 and logical_step == args.stop_logical_step:
                        print('early stop at logical_step %d' % (logical_step))
                        return

            train_dataset.reset()
            print('epoch #%d finished' % (i + 1))
