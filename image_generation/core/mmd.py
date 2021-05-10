'''
MMD functions implemented in tensorflow.
'''
from __future__ import division

import tensorflow as tf
import numpy as np
from .ops import dot, sq_sum, _eps, squared_norm_jacobian
slim = tf.contrib.slim
import math
mysqrt = lambda x: tf.sqrt(tf.maximum(x + _eps, 0.))
from core.snops import linear, lrelu

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel


def any_function(name, z_, dim_f, layer_shape=[64, 64, 64], std=0.25, reuse_=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse_):
        input_shape = z_.get_shape().as_list()
        output_shape = input_shape
        output_ = tf.reshape(z_, (-1, input_shape[-1]))
        output_shape[-1] = dim_f

        for i in range(len(layer_shape)):
            output_ = slim.fully_connected(output_, layer_shape[i], activation_fn=tf.nn.relu,
                                 scope='omega_%i'%i)
            # output_ = tf.nn.relu(linear(output_, name='omega_%i'%i,  output_size=layer_shape[i], with_sn=True, scale=1.))
        out = slim.fully_connected(output_, dim_f, activation_fn=None,
                           scope='omega_last')
        # out = linear(output_, name='omega_last', output_size=dim_f, with_sn=True, scale=1.)

        out = tf.reshape(out, output_shape)

    return out


# there are some redundancy, I'll remove later
def actual_imp_fn(X, Y, imp_type, noise_num=1000, noise_dim=32, tile_num=64,
                  reg_ratio=1.0):  # actual implementation of implicit kernel
    z_1 = tf.expand_dims(X, 0)
    z_2 = tf.expand_dims(Y, 0)
    z_dim = z_1.get_shape().as_list()[-1]
    noise_dim = max(4, 2 * z_dim)
    layer_ratio = max(4, int(16 / z_dim))  # 16
    noise_z_init = tf.random_uniform(shape=[noise_num, noise_dim], minval=-1., maxval=1.)
    # noise_z_init = tf.random_normal(shape=[noise_num, noise_dim])
    #############################
    noise_z = any_function('k_net/omega_1', tf.expand_dims(noise_z_init, 0), z_dim,
                           layer_shape=[layer_ratio * z_dim, layer_ratio * z_dim, layer_ratio * z_dim])#, std=0.25)
    noise_z = tf.transpose(noise_z, perm=[0, 2, 1])
    noise_norm_0 = tf.reduce_mean(tf.reduce_sum(tf.square(noise_z), -2))
    noise_z_0 = tf.random_uniform(shape=[1, 1, noise_num], minval=0, maxval=2 * math.pi)
    z_with_noise_1 = tf.matmul(z_1, noise_z)
    z_with_noise_2 = tf.matmul(z_2, noise_z)
    f_1 = math.sqrt(2) * tf.cos(z_with_noise_1 + noise_z_0)
    f_2 = math.sqrt(2) * tf.cos(z_with_noise_2 + noise_z_0)

    Kxy_1 = tf.matmul(f_1, tf.transpose(f_2, perm=[0, 2, 1])) / noise_num

    f_1_sin = math.sqrt(2) * tf.sin(z_with_noise_1 + noise_z_0)
    f_2_sin = math.sqrt(2) * tf.sin(z_with_noise_2 + noise_z_0)

    Kxy_1_sin = tf.matmul(f_1_sin, tf.transpose(f_2_sin, perm=[0, 2, 1])) / noise_num
    #     f_1_for_scale = tf.sin(z_with_noise_1 + noise_z_0)
    #     f_2_for_scale = tf.sin(z_with_noise_2 + noise_z_0)
    #     Kxy_1_for_scale = tf.matmul(f_1_for_scale, tf.transpose(f_2_for_scale, perm=[0, 2, 1]))/noise_num
    #     part_1_for_scale = tf.reduce_mean(tf.matrix_diag_part(Kxy_1_for_scale))
    #################
    if imp_type == 'imp_2':
        z_1_tile = tf.tile(tf.expand_dims(z_1, 2), (1, 1, tile_num, 1))
        z_2_tile = tf.tile(tf.expand_dims(z_2, 1), (1, tile_num, 1, 1))

        z_1_2_tile = tf.concat([z_1_tile, z_2_tile], axis=-1)
        z_2_1_tile = tf.concat([z_2_tile, z_1_tile], axis=-1)
        z_stage = any_function('k_net/omega_2', z_1_2_tile, 2 * z_dim,
                               layer_shape=[layer_ratio * z_dim, layer_ratio * z_dim, layer_ratio * z_dim])#, std=0.05)
        z_stage += any_function('k_net/omega_2', z_2_1_tile, 2 * z_dim,
                                layer_shape=[layer_ratio * z_dim, layer_ratio * z_dim, layer_ratio * z_dim])#, std=0.05)

        mu = z_stage[:, :, :, :z_dim]
        # logv = tf.nn.tanh(z_stage[:, :, :, z_dim:])
        logv = z_stage[:, :, :, z_dim:]
        '''
        noise_gaussian = tf.random_normal(shape=(noise_num, 1, 1, 1, 1))
        noise_trans = tf.transpose(noise_gaussian, perm=[1, 2, 3, 4, 0])
        mu_z_1 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_1_tile, -1))
        mu_z_2 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_2_tile, -1))
        logv_z_1 = tf.matmul(tf.expand_dims(tf.exp(logv / 2.0), -2), tf.expand_dims(z_1_tile, -1))
        logv_z_2 = tf.matmul(tf.expand_dims(tf.exp(logv / 2.0), -2), tf.expand_dims(z_2_tile, -1))
        z_with_noise_1_2 = mu_z_1 + logv_z_1 * noise_trans
        z_with_noise_2_2 = mu_z_2 + logv_z_2 * noise_trans

        '''
        # TEST code for z_dim larger than 1
        noise_gaussian = tf.random_normal(shape=(1, 1, 1, noise_num, z_dim))
        mu_z_1 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_1_tile, -1))
        mu_z_2 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_2_tile, -1))
        logv_z_1 = tf.matmul(tf.expand_dims(tf.exp(logv/2.0), -2)*noise_gaussian, tf.tile(tf.expand_dims(z_1_tile, -1), [1,1,1,1,1]))
        logv_z_2 = tf.matmul(tf.expand_dims(tf.exp(logv/2.0), -2)*noise_gaussian, tf.tile(tf.expand_dims(z_2_tile, -1), [1,1,1,1,1]))
        z_with_noise_1_2 = mu_z_1 + tf.transpose(logv_z_1, perm=[0,1,2,4,3])
        z_with_noise_2_2 = mu_z_2 + tf.transpose(logv_z_2, perm=[0,1,2,4,3])


        omega_samples = tf.tile(tf.expand_dims(mu, 0), (noise_num, 1, 1, 1, 1))
        omega_samples += tf.random_normal(tf.shape(omega_samples)) * tf.expand_dims(tf.exp(logv / 2.0), 0)
        noise_norm_3 = tf.reduce_mean(tf.reduce_sum(tf.square(omega_samples), -1))  # omega_2_x_y
        noise_norm = tf.matrix_diag_part(tf.reduce_sum(tf.square(omega_samples), -1))  # omega_2_x_x
        noise_norm = tf.reduce_mean(noise_norm, [0, 1])

        z_grad = tf.squeeze(
            tf.stack([tf.gradients(omega_samples[:, :, :, :, i], z_1_2_tile)[0] for i in range(z_dim)], axis=-2),
            axis=0)
        noise_norm_1 = noise_norm_3 + reg_ratio * tf.reduce_mean(
            tf.abs((z_grad[:, :, :, :z_dim] - z_grad[:, :, :, z_dim:]))  # *tf.abs((z_1_tile-z_2_tile)*omega_samples)
        ) / noise_num # this is ok when z_dim == 1, but it's not accurate when z_dim > 1, because it's very inefficient to compute full Jacobian, so we use this just as approximation

        f_1_2 = math.sqrt(2) * tf.cos(z_with_noise_1_2 + noise_z_0)
        f_2_2 = math.sqrt(2) * tf.cos(z_with_noise_2_2 + noise_z_0)
        Kxy_2 = tf.matmul(f_1_2, tf.transpose(f_2_2, perm=[0, 1, 2, 4, 3])) / noise_num

        f_1_2_sin = math.sqrt(2) * tf.sin(z_with_noise_1_2 + noise_z_0)
        f_2_2_sin = math.sqrt(2) * tf.sin(z_with_noise_2_2 + noise_z_0)
        Kxy_2_sin = tf.matmul(f_1_2_sin, tf.transpose(f_2_2_sin, perm=[0, 1, 2, 4, 3])) / noise_num

    if imp_type == 'imp_3':  # new implementation of data-dependent kernel, guaranteed to be positive definite
        z_1_tile = tf.tile(tf.expand_dims(z_1, 2), (1, 1, tile_num, 1))
        z_2_tile = tf.tile(tf.expand_dims(z_2, 1), (1, tile_num, 1, 1))

        #####  testing for another data-dependent construction
        z_stage = any_function('k_net/omega_2', z_1_tile, 2 * z_dim,
                               layer_shape=[layer_ratio * z_dim, layer_ratio * z_dim, layer_ratio * z_dim],
                               )#std=0.05)
        z_stage += any_function('k_net/omega_2', z_2_tile, 2 * z_dim,
                                layer_shape=[layer_ratio * z_dim, layer_ratio * z_dim, layer_ratio * z_dim],
                                )#std=0.05)
        #######

        mu = z_stage[:, :, :, :z_dim]
        logv = z_stage[:, :, :, z_dim:]
        '''
        noise_ = tf.random_uniform(shape=[noise_num, 1, 1, 1, 1, 1], minval=-1., maxval=1.)#tf.random_normal(shape=(noise_num, 1, 1, 1, 1))
        noise_trans = tf.transpose(noise_, perm=[1, 2, 3, 4, 0])
        mu_z_1 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_1_tile, -1))
        mu_z_2 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_2_tile, -1))
        logv_z_1 = tf.matmul(tf.expand_dims(tf.exp(logv / 2.0), -2), tf.expand_dims(z_1_tile, -1))
        logv_z_2 = tf.matmul(tf.expand_dims(tf.exp(logv / 2.0), -2), tf.expand_dims(z_2_tile, -1))
        z_with_noise_1_2 = mu_z_1 + logv_z_1 * noise_trans
        z_with_noise_2_2 = mu_z_2 + logv_z_2 * noise_trans
        '''
        # TEST code for z_dim larger than 1
        noise_ = tf.random_uniform(shape=(1, 1, 1, noise_num, z_dim), minval=-1., maxval=1.)
        mu_z_1 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_1_tile, -1))
        mu_z_2 = tf.matmul(tf.expand_dims(mu, -2), tf.expand_dims(z_2_tile, -1))
        logv_z_1 = tf.matmul(tf.expand_dims(tf.exp(logv/2.0), -2)*noise_, tf.expand_dims(z_1_tile, -1))
        logv_z_2 = tf.matmul(tf.expand_dims(tf.exp(logv/2.0), -2)*noise_, tf.expand_dims(z_2_tile, -1))
        z_with_noise_1_2 = mu_z_1 + tf.transpose(logv_z_1, perm=[0,1,2,4,3])
        z_with_noise_2_2 = mu_z_2 + tf.transpose(logv_z_2, perm=[0,1,2,4,3])


        omega_samples = tf.tile(tf.expand_dims(mu, 0), (noise_num, 1, 1, 1, 1))
        omega_samples += tf.random_normal(tf.shape(omega_samples)) * tf.expand_dims(tf.exp(logv / 2.0), 0)

        noise_norm_3 = tf.reduce_mean(tf.reduce_sum(tf.square(omega_samples), -1))  # omega_2_x_y
        noise_norm = tf.matrix_diag_part(tf.reduce_sum(tf.square(omega_samples), -1))  # omega_2_x_x
        noise_norm = tf.reduce_mean(noise_norm, [0, 1])

        noise_norm_1 = noise_norm_3
        noise_norm_1 += reg_ratio*tf.reduce_mean(tf.abs((tf.squeeze(tf.stack([tf.gradients(omega_samples[:, :, :, :, i], z_1_tile)[0] for i in range(z_dim)], axis=-2), axis=0)
                                                - tf.squeeze(tf.stack([tf.gradients(omega_samples[:, :, :, :, i], z_2_tile)[0] for i in range(z_dim)], axis=-2), axis=0)))#*tf.abs((z_1_tile-z_2_tile)*omega_samples)
                                                )/noise_num # this is ok when z_dim == 1, but it's not accurate when z_dim > 1, because it's very inefficient to compute full Jacobian, so we use this just as approximation

        f_1_2 = math.sqrt(2) * tf.cos(z_with_noise_1_2 + noise_z_0)
        f_2_2 = math.sqrt(2) * tf.cos(z_with_noise_2_2 + noise_z_0)
        Kxy_2 = tf.matmul(f_1_2, tf.transpose(f_2_2, perm=[0, 1, 2, 4, 3])) / noise_num

        f_1_2_sin = math.sqrt(2) * tf.sin(z_with_noise_1_2 + noise_z_0)
        f_2_2_sin = math.sqrt(2) * tf.sin(z_with_noise_2_2 + noise_z_0)
        Kxy_2_sin = tf.matmul(f_1_2_sin, tf.transpose(f_2_2_sin, perm=[0, 1, 2, 4, 3])) / noise_num

    if imp_type == 'imp_2' or imp_type == 'imp_3':
        Kxy = tf.squeeze(Kxy_2) + tf.squeeze(Kxy_1)
        Kxy_sin = tf.squeeze(Kxy_2_sin) + tf.squeeze(Kxy_1_sin)
        norm = noise_norm_1 / 2. + noise_norm_0 / 2.
        noise_norm = noise_norm + noise_norm_0
    else:
        Kxy = tf.squeeze(Kxy_1)
        Kxy_sin = tf.squeeze(Kxy_1_sin)
        norm = noise_norm_0
        noise_norm = norm

    return Kxy, norm, noise_norm, Kxy_sin


def _imp_1_kernel(X, Y, K_XY_only=False):  # implicit kernel here
#     noise_z_init = tf.random_normal(shape=[noise_num, noise_dim])
    K_XY, _, _, K_XY_sin = actual_imp_fn(X, Y, 'imp_1')
    if K_XY_only:
        return K_XY
    K_XX, _, _, K_XX_sin = actual_imp_fn(X, X, 'imp_1')
    K_YY, noise_norm, scale_norm, K_YY_sin = actual_imp_fn(Y, Y, 'imp_1')
    return K_XX, K_XY, K_YY, False, noise_norm, scale_norm, K_XX_sin, K_XY_sin, K_YY_sin


def _imp_2_kernel(X, Y, K_XY_only=False, reg_ratio=1.0):  # implicit kernel here
#     noise_z_init = tf.random_normal(shape=[noise_num, noise_dim])
    K_XY, _, _, K_XY_sin = actual_imp_fn(X, Y, 'imp_2', reg_ratio=reg_ratio)
    if K_XY_only:
        return K_XY
    K_XX, _, _, K_XX_sin = actual_imp_fn(X, X, 'imp_2', reg_ratio=reg_ratio)
    K_YY, noise_norm, scale_norm, K_YY_sin = actual_imp_fn(Y, Y, 'imp_2', reg_ratio=reg_ratio)
    return K_XX, K_XY, K_YY, False, noise_norm, scale_norm, K_XX_sin, K_XY_sin, K_YY_sin

def _imp_3_kernel(X, Y, K_XY_only=False, reg_ratio=1.0):  # implicit kernel here
#     noise_z_init = tf.random_normal(shape=[noise_num, noise_dim])
    K_XY, _, _, K_XY_sin = actual_imp_fn(X, Y, 'imp_3', reg_ratio=reg_ratio)
    if K_XY_only:
        return K_XY
    K_XX, _, _, K_XX_sin = actual_imp_fn(X, X, 'imp_3', reg_ratio=reg_ratio)
    K_YY, noise_norm, scale_norm, K_YY_sin = actual_imp_fn(Y, Y, 'imp_3', reg_ratio=reg_ratio)
    return K_XX, K_XY, K_YY, False, noise_norm, scale_norm, K_XX_sin, K_XY_sin, K_YY_sin

def _distance_kernel(X, Y, K_XY_only=False):
    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XY = c(mysqrt(X_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * XY + c(X_sqnorms) + r(Y_sqnorms))

    if K_XY_only:
        return K_XY

    K_XX = c(mysqrt(X_sqnorms)) + r(mysqrt(X_sqnorms)) - mysqrt(-2 * XX + c(X_sqnorms) + r(X_sqnorms))
    K_YY = c(mysqrt(Y_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms))

    return K_XX, K_XY, K_YY, False


def _tanh_distance_kernel(X, Y, K_XY_only=False):
    return _distance_kernel(tf.tanh(X), tf.tanh(Y), K_XY_only=K_XY_only)


def _dot_kernel(X, Y, K_XY_only=False):
    K_XY = tf.matmul(X, Y, transpose_b=True)
    if K_XY_only:
        return K_XY

    K_XX = tf.matmul(X, X, transpose_b=True)
    K_YY = tf.matmul(Y, Y, transpose_b=True)

    return K_XX, K_XY, K_YY, False


def _rbf_kernel(X, Y, sigma=1., wt=1., K_XY_only=False):

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    XYsqnorm = tf.maximum(-2 * XY + c(X_sqnorms) + r(Y_sqnorms), 0.)

    gamma = 1 / (2 * sigma**2)
    K_XY = wt * tf.exp(-gamma * XYsqnorm)

    if K_XY_only:
        return K_XY

    XXsqnorm = tf.maximum(-2 * XX + c(X_sqnorms) + r(X_sqnorms), 0.)
    YYsqnorm = tf.maximum(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms), 0.)

    gamma = 1 / (2 * sigma**2)
    K_XX = wt * tf.exp(-gamma * XXsqnorm)
    K_YY = wt * tf.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, wt


def _mix_rbf_kernel(X, Y, sigmas=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0], wts=None, K_XY_only=False):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0

    XYsqnorm = tf.maximum(-2 * XY + c(X_sqnorms) + r(Y_sqnorms), 0.)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XY += wt * tf.exp(-gamma * XYsqnorm)

    if K_XY_only:
        return K_XY

    XXsqnorm = tf.maximum(-2 * XX + c(X_sqnorms) + r(X_sqnorms), 0.)
    YYsqnorm = tf.maximum(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms), 0.)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * XXsqnorm)
        K_YY += wt * tf.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def _mix_rq_dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=.1)


def _mix_rq_1dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=1.)


def _mix_rq_10dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=10.)


def _mix_rq_01dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=.1)


def _mix_rq_001dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=.01)


def _tanh_mix_rq_kernel(X, Y, K_XY_only=False):
    return _mix_rq_kernel(tf.tanh(X), tf.tanh(Y), K_XY_only=K_XY_only)


def _mix_rq_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False, add_dot=.0):

    if wts is None:
        wts = [1.] * len(alphas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0., 0., 0.

    XYsqnorm = tf.maximum(-2. * XY + c(X_sqnorms) + r(Y_sqnorms), 0.)

    for alpha, wt in zip(alphas, wts):
        logXY = tf.log(1. + XYsqnorm/(2.*alpha))
        K_XY += wt * tf.exp(-alpha * logXY)
    if add_dot > 0:
        K_XY += tf.cast(add_dot, tf.float32) * XY

    if K_XY_only:
        return K_XY

    XXsqnorm = tf.maximum(-2. * XX + c(X_sqnorms) + r(X_sqnorms), 0.)
    YYsqnorm = tf.maximum(-2. * YY + c(Y_sqnorms) + r(Y_sqnorms), 0.)

    for alpha, wt in zip(alphas, wts):
        logXX = tf.log(1. + XXsqnorm/(2.*alpha))
        logYY = tf.log(1. + YYsqnorm/(2.*alpha))
        K_XX += wt * tf.exp(-alpha * logXX)
        K_YY += wt * tf.exp(-alpha * logYY)
    if add_dot > 0:
        K_XX += tf.cast(add_dot, tf.float32) * XX
        K_YY += tf.cast(add_dot, tf.float32) * YY

    # wts = tf.reduce_sum(tf.cast(wts, tf.float32))
    wts = tf.reduce_sum(tf.cast(wts, tf.float32))
    return K_XX, K_XY, K_YY, wts

################################################################################
### Helper functions to compute variances based on kernel matrices


def mmd2(K, biased=False, rep=-1.):
    K_XX, K_XY, K_YY, const_diagonal = K
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal, biased, rep=rep)  # numerics checked at _mmd2 return


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False, rep=-1.):
    m = tf.cast(K_XX.get_shape()[-2], tf.float32)
    n = tf.cast(K_YY.get_shape()[-2], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX)/ (m * m)
                + tf.reduce_sum(K_YY)*(-rep)  / (n * n)
                - 2 * tf.reduce_sum(K_XY)*((-rep + 1.)/2.) / (m * n))
    else:
        if const_diagonal is not False:
            const_diagonal = tf.cast(const_diagonal, tf.float32)
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
                + (tf.reduce_sum(K_YY) - trace_Y)*(-rep) / (n * (n - 1))
                - 2 * tf.reduce_sum(K_XY)*((-rep + 1.)/2.)  / (m * n))

    return mmd2


def mmd2_and_ratio(K, biased=False, min_var_est=_eps, rep=-1):
    K_XX, K_XY, K_YY, const_diagonal = K
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal, biased, min_var_est, rep=rep)


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False,
                    min_var_est=_eps, rep=-1.):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased, rep=rep)
    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est))
    return mmd2, ratio, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False, rep=-1.):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y)*(-rep) / (m * m)
                - 2 * K_XY_sum*((-rep + 1.)/2.)  / (m * m))
    else:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * (m-1))
                + (Kt_YY_sum + sum_diag_Y)*(-rep) / (m * (m-1))
                - 2 * K_XY_sum*((-rep + 1.)/2.)  / (m * m))

    var_est = (
        2 / (m**2 * (m-1)**2) * (
            2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
            sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
            1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    return mmd2, var_est


def diff_polynomial_mmd2_and_ratio(X, Y, Z):
    dim = tf.cast(X.get_shape()[1], tf.float32)
    # TODO: could definitely do this faster
    K_XY = (tf.matmul(X, Y, transpose_b=True) / dim + 1) ** 3
    K_XZ = (tf.matmul(X, Z, transpose_b=True) / dim + 1) ** 3
    K_YY = (tf.matmul(Y, Y, transpose_b=True) / dim + 1) ** 3
    K_ZZ = (tf.matmul(Z, Z, transpose_b=True) / dim + 1) ** 3
    return _diff_mmd2_and_ratio(K_XY, K_XZ, K_YY, K_ZZ, const_diagonal=False)


def diff_polynomial_mmd2_and_ratio_with_saving(X, Y, saved_sums_for_Z):
    dim = tf.cast(X.get_shape()[1], tf.float32)
    # TODO: could definitely do this faster
    K_XY = (tf.matmul(X, Y, transpose_b=True) / dim + 1) ** 3
    K_YY = (tf.matmul(Y, Y, transpose_b=True) / dim + 1) ** 3
    m = tf.cast(K_YY.get_shape()[0], tf.float32)

    Y_related_sums = _get_sums(K_XY, K_YY)

    mmd2_diff, ratio = _diff_mmd2_and_ratio_from_sums(Y_related_sums, saved_sums_for_Z, m)

    return mmd2_diff, ratio, Y_related_sums


def _diff_mmd2_and_ratio(K_XY, K_XZ, K_YY, K_ZZ, const_diagonal=False):
    m = tf.cast(K_YY.get_shape()[0], tf.float32)  # Assumes X, Y, Z are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to explicitly form them
    return _diff_mmd2_and_ratio_from_sums(
        _get_sums(K_XY, K_YY, const_diagonal),
        _get_sums(K_XZ, K_ZZ, const_diagonal),
        m,
        const_diagonal=const_diagonal
    )


def _diff_mmd2_and_ratio_from_sums(Y_related_sums, Z_related_sums, m, const_diagonal=False):
    Kt_YY_sums, Kt_YY_2_sum, K_XY_sums_0, K_XY_sums_1, K_XY_2_sum = Y_related_sums
    Kt_ZZ_sums, Kt_ZZ_2_sum, K_XZ_sums_0, K_XZ_sums_1, K_XZ_2_sum = Z_related_sums

    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    Kt_ZZ_sum = tf.reduce_sum(Kt_ZZ_sums)

    K_XY_sum = tf.reduce_sum(K_XY_sums_0)
    K_XZ_sum = tf.reduce_sum(K_XZ_sums_0)

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...

    ### Estimators for the various terms involved
    muY_muY = Kt_YY_sum / (m * (m-1))
    muZ_muZ = Kt_ZZ_sum / (m * (m-1))

    muX_muY = K_XY_sum / (m * m)
    muX_muZ = K_XZ_sum / (m * m)

    E_y_muY_sq = (sq_sum(Kt_YY_sums) - Kt_YY_2_sum) / (m*(m-1)*(m-2))
    E_z_muZ_sq = (sq_sum(Kt_ZZ_sums) - Kt_ZZ_2_sum) / (m*(m-1)*(m-2))

    E_x_muY_sq = (sq_sum(K_XY_sums_1) - K_XY_2_sum) / (m*m*(m-1))
    E_x_muZ_sq = (sq_sum(K_XZ_sums_1) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muX_sq = (sq_sum(K_XY_sums_0) - K_XY_2_sum) / (m*m*(m-1))
    E_z_muX_sq = (sq_sum(K_XZ_sums_0) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muY_y_muX = dot(Kt_YY_sums, K_XY_sums_0) / (m*m*(m-1))
    E_z_muZ_z_muX = dot(Kt_ZZ_sums, K_XZ_sums_0) / (m*m*(m-1))

    E_x_muY_x_muZ = dot(K_XY_sums_1, K_XZ_sums_1) / (m*m*m)

    E_kyy2 = Kt_YY_2_sum / (m * (m-1))
    E_kzz2 = Kt_ZZ_2_sum / (m * (m-1))

    E_kxy2 = K_XY_2_sum / (m * m)
    E_kxz2 = K_XZ_2_sum / (m * m)

    ### Combine into overall estimators
    mmd2_diff = muY_muY - 2 * muX_muY - muZ_muZ + 2 * muX_muZ

    first_order = 4 * (m-2) / (m * (m-1)) * (
        E_y_muY_sq - muY_muY**2
        + E_x_muY_sq - muX_muY**2
        + E_y_muX_sq - muX_muY**2
        + E_z_muZ_sq - muZ_muZ**2
        + E_x_muZ_sq - muX_muZ**2
        + E_z_muX_sq - muX_muZ**2
        - 2 * E_y_muY_y_muX + 2 * muY_muY * muX_muY
        - 2 * E_x_muY_x_muZ + 2 * muX_muY * muX_muZ
        - 2 * E_z_muZ_z_muX + 2 * muZ_muZ * muX_muZ
    )
    second_order = 2 / (m * (m-1)) * (
        E_kyy2 - muY_muY**2
        + 2 * E_kxy2 - 2 * muX_muY**2
        + E_kzz2 - muZ_muZ**2
        + 2 * E_kxz2 - 2 * muX_muZ**2
        - 4 * E_y_muY_y_muX + 4 * muY_muY * muX_muY
        - 4 * E_x_muY_x_muZ + 4 * muX_muY * muX_muZ
        - 4 * E_z_muZ_z_muX + 4 * muZ_muZ * muX_muZ
    )
    var_est = first_order + second_order

    ratio = mmd2_diff / mysqrt(tf.maximum(var_est, _eps))
    return mmd2_diff, ratio


def _get_sums(K_XY, K_YY, const_diagonal=False):
    m = tf.cast(K_YY.get_shape()[0], tf.float32)  # Assumes X, Y, Z are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to explicitly form them
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_Y = const_diagonal
        sum_diag2_Y = m * const_diagonal**2
    else:
        diag_Y = tf.diag_part(K_YY)

        sum_diag2_Y = sq_sum(diag_Y)

    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y

    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum = sq_sum(K_XY)

    return Kt_YY_sums, Kt_YY_2_sum, K_XY_sums_0, K_XY_sums_1, K_XY_2_sum


def np_diff_polynomial_mmd2_and_ratio_with_saving(X, Y, saved_sums_for_Z):
    dim = float(X.shape[1])
    # TODO: could definitely do this faster
    K_XY = (np.dot(X, Y.transpose()) / dim + 1) ** 3
    K_YY = (np.dot(Y, Y.transpose()) / dim + 1) ** 3
    m = float(K_YY.shape[0])

    Y_related_sums = _np_get_sums(K_XY, K_YY)

    if saved_sums_for_Z is None:
        return Y_related_sums

    mmd2_diff, ratio = _np_diff_mmd2_and_ratio_from_sums(Y_related_sums, saved_sums_for_Z, m)

    return mmd2_diff, ratio, Y_related_sums


def _np_diff_mmd2_and_ratio_from_sums(Y_related_sums, Z_related_sums, m, const_diagonal=False):
    Kt_YY_sums, Kt_YY_2_sum, K_XY_sums_0, K_XY_sums_1, K_XY_2_sum = Y_related_sums
    Kt_ZZ_sums, Kt_ZZ_2_sum, K_XZ_sums_0, K_XZ_sums_1, K_XZ_2_sum = Z_related_sums

    Kt_YY_sum = Kt_YY_sums.sum()
    Kt_ZZ_sum = Kt_ZZ_sums.sum()

    K_XY_sum = K_XY_sums_0.sum()
    K_XZ_sum = K_XZ_sums_0.sum()

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...

    ### Estimators for the various terms involved
    muY_muY = Kt_YY_sum / (m * (m-1))
    muZ_muZ = Kt_ZZ_sum / (m * (m-1))

    muX_muY = K_XY_sum / (m * m)
    muX_muZ = K_XZ_sum / (m * m)

    E_y_muY_sq = (np.dot(Kt_YY_sums, Kt_YY_sums) - Kt_YY_2_sum) / (m*(m-1)*(m-2))
    E_z_muZ_sq = (np.dot(Kt_ZZ_sums, Kt_ZZ_sums) - Kt_ZZ_2_sum) / (m*(m-1)*(m-2))

    E_x_muY_sq = (np.dot(K_XY_sums_1, K_XY_sums_1) - K_XY_2_sum) / (m*m*(m-1))
    E_x_muZ_sq = (np.dot(K_XZ_sums_1, K_XZ_sums_1) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muX_sq = (np.dot(K_XY_sums_0, K_XY_sums_0) - K_XY_2_sum) / (m*m*(m-1))
    E_z_muX_sq = (np.dot(K_XZ_sums_0, K_XZ_sums_0) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muY_y_muX = np.dot(Kt_YY_sums, K_XY_sums_0) / (m*m*(m-1))
    E_z_muZ_z_muX = np.dot(Kt_ZZ_sums, K_XZ_sums_0) / (m*m*(m-1))

    E_x_muY_x_muZ = np.dot(K_XY_sums_1, K_XZ_sums_1) / (m*m*m)

    E_kyy2 = Kt_YY_2_sum / (m * (m-1))
    E_kzz2 = Kt_ZZ_2_sum / (m * (m-1))

    E_kxy2 = K_XY_2_sum / (m * m)
    E_kxz2 = K_XZ_2_sum / (m * m)

    ### Combine into overall estimators
    mmd2_diff = muY_muY - 2 * muX_muY - muZ_muZ + 2 * muX_muZ

    first_order = 4 * (m-2) / (m * (m-1)) * (
        E_y_muY_sq - muY_muY**2
        + E_x_muY_sq - muX_muY**2
        + E_y_muX_sq - muX_muY**2
        + E_z_muZ_sq - muZ_muZ**2
        + E_x_muZ_sq - muX_muZ**2
        + E_z_muX_sq - muX_muZ**2
        - 2 * E_y_muY_y_muX + 2 * muY_muY * muX_muY
        - 2 * E_x_muY_x_muZ + 2 * muX_muY * muX_muZ
        - 2 * E_z_muZ_z_muX + 2 * muZ_muZ * muX_muZ
    )
    second_order = 2 / (m * (m-1)) * (
        E_kyy2 - muY_muY**2
        + 2 * E_kxy2 - 2 * muX_muY**2
        + E_kzz2 - muZ_muZ**2
        + 2 * E_kxz2 - 2 * muX_muZ**2
        - 4 * E_y_muY_y_muX + 4 * muY_muY * muX_muY
        - 4 * E_x_muY_x_muZ + 4 * muX_muY * muX_muZ
        - 4 * E_z_muZ_z_muX + 4 * muZ_muZ * muX_muZ
    )
    var_est = first_order + second_order

    ratio = mmd2_diff / np.sqrt(max(var_est, _eps))
    return mmd2_diff, ratio


def _np_get_sums(K_XY, K_YY, const_diagonal=False):
    m = float(K_YY.shape[0])  # Assumes X, Y, Z are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to explicitly form them
    if const_diagonal is not False:
        const_diagonal = float(const_diagonal)
        diag_Y = const_diagonal
        sum_diag2_Y = m * const_diagonal**2
    else:
        diag_Y = np.diag(K_YY)

        sum_diag2_Y = np.dot(diag_Y, diag_Y)

    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y

    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    K_XY_2_sum = (K_XY ** 2).sum()

    return Kt_YY_sums, Kt_YY_2_sum, K_XY_sums_0, K_XY_sums_1, K_XY_2_sum
