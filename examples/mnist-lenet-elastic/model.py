import mindspore as ms

from mindspore.common.initializer import Normal


class LeNet5(ms.nn.Cell):
    """Lenet network structure."""

    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = ms.nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = ms.nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = ms.nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = ms.nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = ms.nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = ms.nn.ReLU()
        self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = ms.nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
