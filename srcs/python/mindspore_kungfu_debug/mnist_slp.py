import mindspore as ms

from mindspore.ops import operations as P


class MnistSLP(ms.nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        """Define the operator required."""
        super(MnistSLP, self).__init__()
        weight_init = ms.common.initializer.Normal(0.02)
        # bias_init = ms.common.initializer.Normal(0.02)
        self.fc = ms.nn.Dense(
            32 * 32,
            num_class,
            weight_init=weight_init,
        )
        self.reshape = P.Reshape()

    def construct(self, x):
        """Use the preceding operators to construct networks."""
        bs = x.shape[0]
        x = self.reshape(x, (bs, 32 * 32))
        x = self.fc(x)
        return x
