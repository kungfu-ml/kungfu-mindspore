import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
from .momentum import Momentum
from .sgd import SGD


class KungFuMomentum(Momentum):
    def __init__(self, *args, **kwargs):
        super(KungFuMomentum, self).__init__(*args, **kwargs)
        self.map_ = ms.ops.composite.Map()
        self.all_reduce = kfops.KungFuAllReduce()

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        return super(KungFuMomentum, self).construct(gradients)


class KungFuSGD(SGD):
    def __init__(self, *args, **kwargs):
        super(KungFuSGD, self).__init__(*args, **kwargs)
        self.map_ = ms.ops.composite.Map()
        self.all_reduce = kfops.KungFuAllReduce()

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        return super(KungFuSGD, self).construct(gradients)


def build_optimizer(args, net):
    Optimizer = KungFuMomentum
    opt = Optimizer(
        [x for x in net.get_parameters() if x.requires_grad],
        args.learning_rate,
        args.momentum,
    )
    return opt
