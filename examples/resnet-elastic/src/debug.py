import time

import mindspore as ms


class DebugStopHook(ms.train.callback.Callback):
    def __init__(self, stop_after=1):
        self.stop_after = stop_after
        self.step = 0

    def begin(self, run_context):
        pass

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context):
        pass

    def step_begin(self, run_context):
        self.step += 1

    def step_end(self, run_context):
        if self.step >= self.stop_after:
            run_context.request_stop()
            print('requested stop')

    def end(self, run_context):
        print('stopped')


class LogStepCallback(ms.train.callback.Callback):
    def __init__(self):
        self.step = 0
        self._last = time.time()

    def begin(self, run_context):
        print('%s::%s' % ('LogStepCallback', 'begin'))

    def epoch_begin(self, run_context):
        print('%s::%s' % ('LogStepCallback', 'epoch_begin'))

    def epoch_end(self, run_context):
        print('%s::%s' % ('LogStepCallback', 'epoch_end'))

    def step_begin(self, run_context):
        self.step += 1

    def step_end(self, run_context):
        t1 = time.time()
        d = t1 = self._last
        print('step took %.3fms' % (d * 1e3))
        self._last = t1

    def end(self, run_context):
        print('stopped')
