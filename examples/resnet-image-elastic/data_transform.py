# import collections
# from io import BytesIO
# import os
# import mindspore.dataset as ds
from mindspore.mindrecord import TFRecordToMR
# import mindspore.dataset.vision.c_transforms as vision
# from PIL import Image
import tensorflow as tf

import os
import mindspore.common.dtype as mstype
import mindspore.dataset as msds
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size


tf_dir = '/data/imagenet/records/'
ms_dir = '/data/imagenet/ms_record/'

def transform_single(file):
    tf_filename = os.path.join(tf_dir, file)
    filenames = [tf_filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    image_feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }

    def _parse_image_function(example_proto):
    # parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_dataset.map(_parse_image_function)

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def image_example(image_string, label):
        label = label.numpy().astype(int)
        feature = {
        'label': _int64_feature(label),
        'image': _bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    record_file = os.path.join(ms_dir, file + str('-tf'))
    ms_filename = os.path.join(ms_dir, file)
    with tf.io.TFRecordWriter(record_file) as writer:
        for image_features in parsed_image_dataset:
            label = image_features['image/class/label']
            image = image_features['image/encoded']
            tf_example = image_example(image, label)
            writer.write(tf_example.SerializeToString())
        

    feature_dict = {"image": tf.io.FixedLenFeature([], tf.string),
                    "label": tf.io.FixedLenFeature([], tf.int64)
                }

    tfrecord_transformer = TFRecordToMR(record_file, ms_filename, feature_dict, ["image"])
    tfrecord_transformer.transform()
    os.remove(record_file)

for file in os.listdir(tf_dir):
    print(f'working on {file}...')
    transform_single(file)
