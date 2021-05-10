import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt


input_dir = input('input data directory please: ') # input_dir/*.png ...
output_dir = os.path.join(input_dir, "tf_records_train")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def _int64_feature(value):
    """
    generate int64 feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    generate byte feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def createRecord(imageDir):
    """
    create TFRecord data.

    Arguments:
    imageDir -- image directory.
    Return: none.
    """
    writer = tf.python_io.TFRecordWriter(os.path.join(imageDir, "tf_records_train/train"))
    # classNames = ["cat", "dog", "horse"]

    # for classIndex, className in enumerate(classNames):
    #     print "class name = ",className
    #     currentClassDir = os.path.join(imageDir,className)
    #     print "current dir = ",currentClassDir
    count = 0
    for index, imageName in enumerate(os.listdir(imageDir)):
        if not os.path.isdir(os.path.join(imageDir,imageName)):
            image = Image.open(os.path.join(imageDir,imageName))
            img_shape=np.shape(image)
            print(img_shape)
            if img_shape[-1] == 3:
                image_raw = image.tobytes() # convert image to binary format
                print ('nice, ', index, imageName)

                example = tf.train.Example(features = tf.train.Features(feature = {
                "image/height": _int64_feature(img_shape[0]),
                "image/width": _int64_feature(img_shape[1]),
                "image/encoded": _bytes_feature(image_raw),
                }))
                writer.write(example.SerializeToString())
            count += 1
    writer.close()

createRecord(input_dir)
