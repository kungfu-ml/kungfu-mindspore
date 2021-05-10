# minimize y = x * x
import argparse

import mindspore as ms
import numpy as np


class LogStepHook(ms.train.callback.Callback):
    def __init__(self):
        self._epoch = 0
        self._step = 0

    def begin(self, run_context):
        print('%s::%s' % ('LogStepHook', 'BEGIN'))

    def epoch_begin(self, run_context):
        print('epoch begin %d' % (self._epoch))

    def epoch_end(self, run_context):
        print('epoch end %d' % (self._epoch))
        self._epoch += 1

    def step_begin(self, run_context):
        print('step begin %d' % (self._step))

    def step_end(self, run_context):
        print('step end %d' % (self._step))
        self._step += 1

    def end(self, run_context):
        print('%s::%s, trained %d steps, %d epochs' %
              ('LogStepHook', 'END', self._step, self._epoch))


class QuadraticModel(ms.train.model.Model):
    def __init__(self, *args, **kwargs):
        print('%s::%s' % ('QuadraticModel', '__init__'))
        super(QuadraticModel, self).__init__(*args, **kwargs)
        # TODO: auto inject kungfu callback

    def train(self, *args, **kwargs):
        print('%s::%s' % ('QuadraticModel', 'train'))
        # TODO: auto inject kungfu callback
        super(QuadraticModel, self).train(*args, **kwargs)


def create_dataset(size):
    np.random.seed(58)
    data = np.random.sample((size, 2))
    label = np.random.sample((size, 1))

    def GeneratorFunc():
        for i in range(size):
            yield (data[i], label[i])

    dataset = ms.dataset.GeneratorDataset(
        GeneratorFunc,
        ["data", "label"],
    )
    return dataset


class IdentityLoss(ms.nn.Cell):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def construct(self, x, y):
        # print('%s :: (%s, %s)' % ('IdentityLoss', x, y)) # can't print
        return x


class QuadraticFunction(ms.nn.Cell):
    def __init__(self):
        super(QuadraticFunction, self).__init__()
        init = ms.common.initializer.Normal(0.02)
        shape = [1]

        self.w = ms.common.parameter.Parameter(
            ms.common.initializer.initializer(init, shape),  #
        )

    def construct(self, x):
        return self.w * self.w


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default='CPU',
                   choices=['Ascend', 'GPU', 'CPU'])
    p.add_argument('--learning-rate', type=float, default=0.01)
    p.add_argument('--epochs', type=int, default=1)
    return p.parse_args()


def log_args(args):
    print('device=%s' % (args.device))


def main():
    args = parse_args()
    log_args(args)
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device,
                           save_graphs=False)

    net = QuadraticFunction()

    # opt = ms.nn.Momentum(
    #     [x for x in net.get_parameters() if x.requires_grad],
    #     args.learning_rate,
    #     args.momentum,
    # )
    opt = ms.nn.SGD(
        [x for x in net.get_parameters() if x.requires_grad],
        args.learning_rate,
        # args.momentum,
    )

    loss = IdentityLoss()
    # loss = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True,
    #                                                 reduction='mean')

    model = QuadraticModel(
        net,
        loss_fn=loss,
        optimizer=opt,
        # loss_scale_manager=loss_scale,
        metrics={'acc'},
        # amp_level="O2",
        keep_batchnorm_fp32=False,
    )
    print(model)

    dataset = create_dataset(10)
    sink_mode = False
    model.train(
        epoch=args.epochs,
        train_dataset=dataset,
        callbacks=[
            LogStepHook(),
        ],
        dataset_sink_mode=sink_mode,
    )


main()
