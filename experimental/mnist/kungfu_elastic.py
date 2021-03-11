import mindspore as ms

NAME = 'kungfu-elastic'
__version__ = '0.0.0'


def info():
    print('%s ,version: %s' % (NAME, __version__))


class Config:
    def __init__(self, batch_size, schedule=None):
        self._batch_size = batch_size
        self._schedule = schedule


class KungFuModel(ms.train.model.Model):  # replace ms.train.model.Model
    def __init__(self, *args, **kwargs):
        print('%s::%s' % ('KungFuModel', '__init__'))
        super(KungFuModel, self).__init__(*args, **kwargs)
        # TODO: auto inject kungfu callback

    def train(self, *args, **kwargs):
        print('%s::%s' % ('KungFuModel', 'train'))
        # TODO: auto inject kungfu callback
        super(KungFuModel, self).train(*args, **kwargs)
