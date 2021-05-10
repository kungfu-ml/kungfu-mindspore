from mindspore_kungfu_debug.resnet import resnet50, ResNet, ResidualBlock


def resnet(class_num):
    return ResNet(
        block=ResidualBlock,
        # layer_nums=[3, 4, 6, 3],
        layer_nums=[3, 4, 1, 3],
        in_channels=[64, 256, 512, 1024],
        out_channels=[256, 512, 1024, 2048],
        strides=[1, 2, 2, 2],
        num_classes=class_num,
    )


def model_fn():
    return resnet(class_num=10)
