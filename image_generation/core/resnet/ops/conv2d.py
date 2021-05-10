
import tensorflow as tf
from ... import sn

default_weightnorm = False


def enable_default_weightnorm():
    global default_weightnorm
    default_weightnorm = True


def Conv2D(name, input_dim, output_dim, filter_size, inputs, stride=1, biases=True, with_sn=False, with_learnable_sn_scale=False, update_collection=None, scale=1.):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.variable_scope(name):

        # print "WARNING IGNORING GAIN"
        strides = [1, 1, stride, stride]
        w = tf.get_variable('w', [filter_size, filter_size, input_dim, output_dim],
                            initializer=tf.glorot_uniform_initializer())
        if with_sn:
            s = tf.get_variable('s', shape=[1], initializer=tf.constant_initializer(scale), trainable=with_learnable_sn_scale, dtype=tf.float32)
            w_bar, sigma = sn.spectral_normed_weight(w, update_collection=update_collection, with_sigma=True)
            w_bar = s*w_bar
            conv = tf.nn.conv2d(inputs, w_bar, strides=strides, padding='SAME', data_format='NCHW')
        else:
            conv = tf.nn.conv2d(inputs, w, strides=strides, padding='SAME', data_format='NCHW')

        if biases:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases, data_format='NCHW'), conv.get_shape())

        return conv
