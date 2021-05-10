import mindspore as ms

opts = [d for d in dir(ms.nn.optim)]
assert ('KungFuSGD' in opts)
assert ('KungFuMomentum' in opts)
