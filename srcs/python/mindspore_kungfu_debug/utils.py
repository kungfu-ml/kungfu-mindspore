import time
import os
import numpy as np

import mindspore as ms


def measure(f):
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    return duration, result


def show_duration(duration):
    if duration < 1:
        return '%.2fms' % (duration * 1e3)
    if duration < 60:
        return '%.2fs' % duration
    sec = int(duration)
    mm, ss = sec / 60, sec % 60
    if duration < 3600:
        return '%dm%ds' % (mm, ss)
    return '%dh%dm%ds' % (mm / 60, mm % 60, ss)


def log_duration(f, *args, **kwargs):
    duration, result = measure(lambda: f(*args, **kwargs))
    print('%s took %s' % (f, show_duration(duration)))
    return result


def save_npz(net, filename):
    # print('save_npz::%s' % (filename))
    values = dict()
    for p in net.get_parameters():
        name = p.name
        v = p.asnumpy()
        values[name] = v
    np.savez(filename, **values)


def save_npz_per_weight(net, name_fn):
    for p in net.get_parameters():
        name = p.name
        v = p.asnumpy()
        filename = name_fn(name)
        values = {name: v}
        np.savez(filename, **values)


def save_data_npz(x, y_, filename):
    np.savez(filename, samples=x.asnumpy(), labels=y_.asnumpy())


def dump_model(net):
    for idx, p in enumerate(net.get_parameters()):
        print('[%d] %s' % (idx, p.name))


def load_ckpt(net, ckpt_name):
    param_dict = ms.train.serialization.load_checkpoint(ckpt_name)
    # for idx, (k, v) in enumerate(param_dict.items()):
    #     print('[%d] %s=%s' % (idx, k, v))
    ms.train.serialization.load_param_into_net(net, param_dict)


def get_ckpt_file_name(args, step, suffix='ckpt'):
    filename = '%s-%06d.%s' % (args.ckpt_prefix, step, suffix)
    return os.path.join(args.ckpt_dir, filename)
