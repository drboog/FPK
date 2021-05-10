
from ... import resnet as lib
import numpy as np
import tensorflow as tf

def Batchnorm(name, axes, inputs): # the data format needs to be NCHW
    axis = 1
    result = tf.layers.batch_normalization(inputs,
                    momentum=0.9, 
                    epsilon=1e-5,
                    scale=True,
                    training=True,
                    fused=True,
                    axis=axis,
                    name=name)
    return result
