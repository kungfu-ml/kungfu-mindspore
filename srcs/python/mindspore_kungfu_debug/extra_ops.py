import mindspore as ms


# FIXME: use official Mod OP
class ModOp(ms.nn.Cell):
    def __init__(self):
        super(ModOp, self).__init__()
        self.div_op = ms.ops.operations.math_ops.Div()

    def construct(self, x, y):
        q = self.div_op(x, y)
        r = x - q * y
        return r
