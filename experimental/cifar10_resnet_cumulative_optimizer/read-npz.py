import numpy as np

filename = 'checkpoint/cifar10-resnet50-000000.npz'

a = np.load(filename)
w = max(len(k) for k in a)

for k in a:
    v = a[k]
    print('%-*s :: %s %s' % (w, k, v.dtype, v.shape))
