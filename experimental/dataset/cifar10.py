import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C
from mindspore.communication.management import get_group_size, get_rank, init


def create_dataset1(dataset_path,
                    do_train,
                    repeat_num=1,
                    batch_size=32,
                    target="Ascend"):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        init()
        rank_id = get_rank()
        device_num = get_group_size()

    if device_num == 1:
        ds = de.Cifar10Dataset(dataset_path,
                               num_parallel_workers=8,
                               shuffle=True)
    else:
        ds = de.Cifar10Dataset(dataset_path,
                               num_parallel_workers=8,
                               shuffle=True,
                               num_shards=device_num,
                               shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(operations=type_cast_op,
                input_columns="label",
                num_parallel_workers=8)
    ds = ds.map(operations=trans,
                input_columns="image",
                num_parallel_workers=8)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds
