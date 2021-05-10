from tensorflow.python.framework import ops
from utils.misc import variable_summaries
from .mmd import tf
try:
    from .sn import spectral_normed_weight
except:
    from sn import spectral_normed_weight


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm", format='NCHW'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
            if format == 'NCHW':
                self.axis = 1
            elif format == 'NHWC':
                self.axis = 3

    def __call__(self, x, train=True):
        # return tf.contrib.layers.batch_norm(x,
        #                   decay=self.momentum,
        #                   updates_collections=tf.GraphKeys.UPDATE_OPS,
        #                   epsilon=self.epsilon,
        #                   scale=True,
        #                   is_training=train,
        #                   fused=True,
        #                   data_format=self.format,
        #                   scope=self.name)

        return tf.layers.batch_normalization(
            x,
            momentum=self.momentum,
            epsilon=self.epsilon,
            scale=True,
            training=train,
            fused=True,
            axis=self.axis,
            name=self.name)


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, scale=1.0, with_learnable_sn_scale=False, with_sn=False,
           name="snconv2d",  update_collection=None, data_format='NCHW',with_singular_values=False):
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       tf.get_variable_scope().name)
        has_summary = any([('w' in v.op.name) for v in scope_vars])
        out_channel, in_channel = get_in_out_shape([output_dim], input_.get_shape().as_list(), data_format)
        strides = get_strides(d_h, d_w, data_format)
        w = tf.get_variable('w', [k_h, k_w, in_channel, out_channel],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        if with_sn:
            s = tf.get_variable('s', shape=[1], initializer=tf.constant_initializer(scale), trainable=with_learnable_sn_scale, dtype=tf.float32)
            w_bar, sigma = spectral_normed_weight(w, update_collection=update_collection, with_sigma=True)
            w_bar = s*w_bar
            conv = tf.nn.conv2d(input_, w_bar, strides=strides, padding='SAME', data_format=data_format)
        else:
            conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME', data_format=data_format)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases, data_format=data_format), conv.get_shape())

        if not has_summary:
            if with_sn:
                variable_summaries({ 'b': biases, 's': s, 'sigma_w': sigma})
                variable_summaries({'W': w},with_singular_values=with_singular_values)
            else:
                variable_summaries({'b': biases})
                variable_summaries({'W': w},with_singular_values=with_singular_values)


        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, scale=1.0, with_learnable_sn_scale=False, with_sn=False,
             name="deconv2d", with_w=False, update_collection=None, data_format='NCHW',with_singular_values=False):
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       tf.get_variable_scope().name)
        has_summary = any([('w' in v.op.name) for v in scope_vars])
        out_channel, in_channel = get_in_out_shape(output_shape, input_.get_shape().as_list(), data_format)
        strides = get_strides(d_h, d_w, data_format)
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, out_channel, in_channel],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        if with_sn:
            s = tf.get_variable('s', shape=[1], initializer=tf.constant_initializer(scale), trainable=with_learnable_sn_scale, dtype=tf.float32)
            w_bar, sigma = spectral_normed_weight(w, update_collection=update_collection, with_sigma=True)
            w_bar = s*w_bar
            deconv = tf.nn.conv2d_transpose(input_, w_bar, output_shape=output_shape, strides=strides, data_format=data_format)
        else:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=strides, data_format=data_format)

        biases = tf.get_variable('biases', [out_channel], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases, data_format=data_format), deconv.get_shape())

        if not has_summary:
            if with_sn:
                variable_summaries({ 'b': biases, 's': s, 'sigma_w': sigma})
                variable_summaries({'W': w},with_singular_values=with_singular_values)
            else:
                variable_summaries({'b': biases})
                variable_summaries({'W': w},with_singular_values=with_singular_values)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def get_in_out_shape(output_shape, input_shape, format):
    if format == 'NCHW':
        if len(output_shape) > 1:
            out_channel = output_shape[1]
        else:
            out_channel = output_shape[0]
        in_channel = input_shape[1]
    elif format == 'NHWC':
        if len(output_shape) > 1:
            out_channel = output_shape[-1]
        else:
            out_channel = output_shape[0]
        in_channel = input_shape[-1]
    return out_channel, in_channel


def get_strides(d_h, d_w, format):
    if format == 'NCHW':
        return [1, 1, d_h, d_w]
    elif format == 'NHWC':
        return [1, d_h, d_w, 1]


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, name="Linear", stddev=0.01, scale=1.0, with_learnable_sn_scale=False, with_sn=False, bias_start=0.0, with_w=False, update_collection=None, with_singular_values=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       tf.get_variable_scope().name)
        has_summary = any([('Matrix' in v.op.name) for v in scope_vars])
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

        if with_sn:
            s = tf.get_variable('s', shape=[1], initializer=tf.constant_initializer(scale), trainable=with_learnable_sn_scale, dtype=tf.float32)

            matrix_bar, sigma = spectral_normed_weight(matrix, update_collection=update_collection, with_sigma=True)
            matrix_bar = s*matrix_bar
            mul = tf.matmul(input_, matrix_bar)

        else:
            mul = tf.matmul(input_, matrix)

        bias = tf.get_variable(
            "bias",
            [output_size],
            initializer=tf.constant_initializer(bias_start))

        if not has_summary:
            if with_sn:
                variable_summaries({'b': bias, 's': s, 'sigma_w': sigma})
                variable_summaries({'W': matrix}, with_singular_values=with_singular_values) 
            else:
                variable_summaries({'b': bias})
                variable_summaries({'W': matrix}, with_singular_values=with_singular_values) 

        if with_w:
            return mul + bias, matrix, bias
        else:
            return mul + bias


def linear_one_hot(input_, output_size, num_classes, name="Linear_one_hot", stddev=0.01, scale=1.0, with_learnable_sn_scale=False, with_sn=False, bias_start=0.0, with_w=False, update_collection=None,with_singular_values=False):
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       tf.get_variable_scope().name)
        has_summary = any([('Matrix' in v.op.name) for v in scope_vars])
        matrix = tf.get_variable(
            "Matrix",
            [num_classes, output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))

        if with_sn:
            s = tf.get_variable('s', shape=[1], initializer=tf.constant_initializer(scale), trainable=with_learnable_sn_scale, dtype=tf.float32)

            matrix_bar, sigma = spectral_normed_weight(matrix, update_collection=update_collection, with_sigma=True)
            matrix_bar = s*matrix_bar
            embed = tf.nn.embedding_lookup(matrix_bar, input_)

        else:
            embed = tf.nn.embedding_lookup(matrix, input_)

        if not has_summary:
            if with_sn:
                variable_summaries({'s': s, 'sigma_w': sigma})
                variable_summaries({'W': matrix}, with_singular_values=with_singular_values) 
            else:
                variable_summaries({'W': matrix}, with_singular_values=with_singular_values) 

        if with_w:
            return embed, matrix
        else:
            return embed
