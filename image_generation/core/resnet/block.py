
import functools
import tensorflow as tf
from core.resnet.ops import conv2d, batchnorm, layernorm, cond_batchnorm

def ResidualBlockNew(name, input_dim, output_dim, filter_size, inputs, y=None, num_classes=None, resample=None, he_init=True, mode='', with_sn=False, with_learnable_sn_scale=False, update_collection=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = MeanPoolConv
        conv_1 = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = UpsampleConv
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = conv2d.Conv2D
        conv_1 = functools.partial(conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2 = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample is None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(
            name+'.Shortcut',
            input_dim=input_dim,
            output_dim=output_dim,
            filter_size=1,
            biases=True,
            inputs=inputs,
            with_sn=False,
            with_learnable_sn_scale=with_learnable_sn_scale,
            update_collection=update_collection)

    output = inputs
    output = Normalize(name+'.BN1', [0, 2, 3], output, mode=mode, labels=y, num_classes=num_classes)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, biases=False, with_sn=with_sn, with_learnable_sn_scale=with_learnable_sn_scale, update_collection=update_collection)
    output = Normalize(name+'.BN2', [0, 2, 3], output, mode=mode, labels=y, num_classes=num_classes)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, with_sn=with_sn, with_learnable_sn_scale=with_learnable_sn_scale, update_collection=update_collection)

    return shortcut + output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, y=None, num_classes=None, resample=None, he_init=True, mode='', with_sn=False, with_learnable_sn_scale=False, update_collection=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = MeanPoolConv
        conv_1 = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = UpsampleConv
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = conv2d.Conv2D
        conv_1 = functools.partial(conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2 = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample is None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(
            name+'.Shortcut',
            input_dim=input_dim,
            output_dim=output_dim,
            filter_size=1,
            biases=True,
            inputs=inputs,
            with_sn=with_sn,
            with_learnable_sn_scale=with_learnable_sn_scale,
            update_collection=update_collection)

    output = inputs
    output = Normalize(name+'.BN1', [0, 2, 3], output, mode=mode, labels=y, num_classes=num_classes)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, biases=False, with_sn=with_sn, with_learnable_sn_scale=with_learnable_sn_scale, update_collection=update_collection)
    output = Normalize(name+'.BN2', [0, 2, 3], output, mode=mode, labels=y, num_classes=num_classes)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, with_sn=with_sn, with_learnable_sn_scale=with_learnable_sn_scale, update_collection=update_collection)

    return shortcut + output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, biases=True, with_sn=False, with_learnable_sn_scale=False, update_collection=None):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, biases=biases, with_sn=with_sn, with_learnable_sn_scale=with_learnable_sn_scale, update_collection=update_collection)
    return output


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, biases=True, with_sn=False, with_learnable_sn_scale=False, update_collection=None):
    output = conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, biases=biases, with_sn=with_sn, with_learnable_sn_scale=with_learnable_sn_scale, update_collection=update_collection)
    output = tf.add_n([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, biases=True, with_sn=False, with_learnable_sn_scale=False, update_collection=None):
    output = inputs
    output = tf.add_n([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, biases=biases, with_sn=with_sn, with_learnable_sn_scale=with_learnable_sn_scale, update_collection=update_collection)
    return output


def Normalize(name, axes, inputs, mode='batchnorm', labels=None, num_classes=0):
    if mode == 'layernorm':
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return layernorm.Layernorm(name, [1, 2, 3], inputs)
    elif mode == 'batchnorm':
        return batchnorm.Batchnorm(name, axes, inputs)
    elif mode == 'cond_batchnorm':
        return cond_batchnorm.Batchnorm(name, axes, inputs, labels=labels, n_labels=num_classes)
    else:
        return inputs
