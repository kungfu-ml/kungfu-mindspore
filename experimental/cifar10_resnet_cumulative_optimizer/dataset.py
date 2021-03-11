import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(data_path, batch_size):
    ds = de.Cifar10Dataset(
        data_path,
        num_parallel_workers=8,
        shuffle=False,
    )

    # define map operations
    trans = []
    # if do_train:
    #     trans += [
    #         # C.RandomCrop((32, 32), (4, 4, 4, 4)),
    #         # C.RandomHorizontalFlip(prob=0.5)
    #     ]

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

    return ds
