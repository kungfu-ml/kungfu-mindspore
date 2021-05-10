import mindspore as ms


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
