import mindspore as ms
import mindspore.nn as nn

from mindspore.ops.operations.kungfu_comm_ops import KungFuLogTensor


def _bn(channel):
    return nn.BatchNorm2d(channel,
                          eps=1e-4,
                          momentum=0.9,
                          gamma_init=1,
                          beta_init=0,
                          moving_mean_init=0,
                          moving_var_init=1)


class LeNet5(ms.nn.Cell):
    def __init__(
        self,
        num_class=10,
        num_channel=3,
        use_bn=False,
        dbg_log_tensor=False,
    ):
        """Define the operator required."""
        super(LeNet5, self).__init__()
        self.use_bn = use_bn
        self.dbg_log_tensor = dbg_log_tensor

        weight_init1 = ms.common.initializer.Normal(0.02)
        weight_init2 = ms.common.initializer.Normal(0.02)
        weight_init3 = ms.common.initializer.Normal(0.02)
        self.conv1 = ms.nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.bn1 = _bn(6)
        self.conv2 = ms.nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.bn2 = _bn(16)
        self.fc1 = ms.nn.Dense(16 * 5 * 5, 120, weight_init=weight_init1)
        self.fc2 = ms.nn.Dense(120, 84, weight_init=weight_init2)
        self.fc3 = ms.nn.Dense(84, num_class, weight_init=weight_init3)
        self.relu = ms.nn.ReLU()
        self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = ms.nn.Flatten()

        self.log_tensor = KungFuLogTensor()

    def construct(self, x):
        """Use the preceding operators to construct networks."""
        if self.dbg_log_tensor:
            x = self.log_tensor(x)

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.max_pool2d(self.relu(x))
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.max_pool2d(self.relu(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # RuntimeError: mindspore/ccsrc/pipeline/jit/validator.cc:63 ValidateOperation] Illegal primitive: Primitive KungFuLogTensor's bprop not defined.
        # if self.dbg_log_tensor:
        #     x = self.log_tensor(x)

        return x
