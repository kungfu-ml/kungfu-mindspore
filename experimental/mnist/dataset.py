import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter


def create_dataset(data_path, batch_size, repeat_num):
    num_parallel_workers = 1
    device_num = 1
    shard_id = 0
    # TODO: replace kungfu dataset
    ds = de.MnistDataset(
        data_path,
        # num_parallel_workers=num_parallel_workers,
        # shuffle=False,
        num_shards=device_num,
        shard_id=shard_id,
    )

    # define map operations
    label_trans = [
        C2.TypeCast(mstype.int32),
    ]

    imgage_trans = [
        # ValueError: For 'MatMul' evaluator shapes of inputs can not do this operator, got 256 and 400,
        # C.Resize((28, 28)),
        CV.Resize((32, 32), interpolation=Inter.LINEAR),  #
        CV.Rescale(1.0 / 255.0, shift=0.0),
        CV.Rescale(1 / 0.3081,
                   shift=-1 * 0.1307 / 0.3081),  # NOT converge if removed
        CV.HWC2CHW()
    ]

    ds = ds.map(operations=label_trans,
                input_columns="label",
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=imgage_trans,
                input_columns="image",
                num_parallel_workers=num_parallel_workers)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)  # repeat_num is deprecated

    return ds
