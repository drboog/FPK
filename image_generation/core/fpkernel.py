from .model import MMD_GAN, tf
from . import mmd
from .ops import safer_norm, tf, squared_norm_jacobian
from .ops import jacob
slim = tf.contrib.slim
import math
class FP(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        self.config = config
        self.batch_size = self.config.batch_size
        super(FP, self).__init__(sess, config, **kwargs)

    def cost_matrix(self, x, y, mode):  # compute the cost matrix (L2 distances)
        if mode == 'l2':
            x_expand = tf.expand_dims(x, axis=-2)
            y_expand = tf.expand_dims(y, axis=-3)
            c = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)  # sum over the dimensions
        elif mode == 'mmd':
            c = self.k_gg + self.k_ii - 2*self.k_gi + self.k_gg_2 + self.k_ii_2 - 2*self.k_gi_2 
        return c 

    def M(self, u, v, eps):
        return (-self.c + tf.expand_dims(u, -2) + tf.expand_dims(v, -1))/eps  # the return shape is (batch_size, batch_size)

    def mmd_loss(self, G, images):
        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            K_XX, K_XY, K_YY, T_or_F, noise_norm, scale_norm, K_XX_sin, K_XY_sin, K_YY_sin = kernel(G, images)
            mmd_loss = mmd.mmd2([K_XX, K_XY, K_YY, T_or_F])

            kernel_2 = getattr(mmd, '_rbf_kernel')
            K_XX_, K_XY_, K_YY_, T_or_F_ = kernel_2(G, images)
            mmd_loss_ = mmd.mmd2([K_XX_, K_XY_, K_YY_, T_or_F_])
            mmd_loss = self.warm_up*mmd_loss + (1 - self.warm_up) * mmd_loss_
            K_XX = self.warm_up*K_XX + (1 - self.warm_up) * K_XX_
            K_XY = self.warm_up*K_XY + (1 - self.warm_up) * K_XY_
            K_YY = self.warm_up*K_YY + (1 - self.warm_up) * K_YY_

            K_XX_sin = self.warm_up*K_XX_sin + (1 - self.warm_up)*K_XX_
            K_YY_sin = self.warm_up*K_YY_sin + (1 - self.warm_up)*K_YY_
            K_XY_sin = self.warm_up*K_XY_sin + (1 - self.warm_up)*K_XY_
            return mmd_loss, K_XX, K_YY, K_XY, noise_norm, scale_norm, K_XX_sin, K_YY_sin, K_XY_sin
        else:
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            K_XX, K_XY, K_YY, T_or_F = kernel(G, images)
            mmd_loss = mmd.mmd2([K_XX, K_XY, K_YY, T_or_F])
            return mmd_loss, K_XX, K_YY, K_XY

    def delete_diag(self, matrix):
        return matrix - tf.matrix_diag(tf.matrix_diag_part(matrix))  # return matrix, while k_ii is 0

    def compute_kde(self, kernel_matrix):  # KDE based on kernel matrix
        kde = tf.reduce_sum(kernel_matrix, axis=-1)/(self.batch_size)
        return kde

    def set_loss(self, D_G, D_images):   # the input here is the output of the discriminator
        # self.z_2 are samples from uniform distribution, self.images_2 are samples from training data
        # G_another = self.generator(self.z_2, self.batch_size)
        self.images_2 = self.batch_queue.dequeue()

        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            mmd_1, k_gg, k_ii, k_gi, noise_norm, scale_norm, k_gg_sin, k_ii_sin, k_gi_sin = self.mmd_loss(D_G, D_images)
            self.k_gg_sin = k_gg_sin
            self.k_ii_sin = k_ii_sin
            self.k_gi_sin = k_gi_sin
        else:
            mmd_1, k_gg, k_ii, k_gi = self.mmd_loss(D_G, D_images)  # k_gg means generated images kernel matrix ...
            self.k_gg_sin = k_gg
            self.k_ii_sin = k_ii
            self.k_gi_sin = k_gi
        self.k_gg = k_gg
        self.k_ii = k_ii
        self.k_gi = k_gi
        self.mmd_1 = mmd_1

        # this is minimizing MMD
        g_loss = self.mmd_1

        # # # this is obtained by stochastic differential equation
        if self.config.train_lan_steps > 0:
            step_lr = self.config.train_lan_step_lr
            noise_std = tf.sqrt(step_lr*2)*0.001
            # noise_std = 0.005
            self.z_2 = self.z
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            d_i = D_images #self.discriminator(self.images,self.real_batch_size, return_layers=False, update_collection="NO_OPS")

            for i in range(self.config.train_lan_steps):
                current_g = self.generator(self.z_2, self.batch_size, update_collection="NO_OPS")
                d_g = self.discriminator(current_g, self.batch_size, return_layers=False, update_collection="NO_OPS")

                # note that we should use k(x,tf.stop_gradient(x)) instead of k(x,x), but k(x,x) also works very well
                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    _, kxy, _, _, _, _, _, _, _ = kernel(d_g, d_i, reg_ratio=self.config.reg_ratio)
                else:
                    _, kxy, _, _, = kernel(d_g, d_i)

                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    _, kxx, _, _, _, _, _, _, _ = kernel(d_g, tf.stop_gradient(d_g), reg_ratio=self.config.reg_ratio)
                else:
                    _, kxx, _, _, = kernel(d_g, tf.stop_gradient(d_g))

                energy = -tf.log(tf.reduce_mean(kxy, axis=-1) + 1e-10) + tf.log(
                    tf.reduce_sum(self.delete_diag(kxx), axis=-1) / (self.config.batch_size - 1) + 1e-10)

                z_grad = tf.gradients(energy, self.z_2)[0]
                # z_grad = tf.clip_by_value(z_grad, -1., 1.)
                self.z_2 = self.z_2 - step_lr*z_grad + tf.random_normal([self.batch_size, self.z_dim], mean=0.,stddev=noise_std, dtype=tf.float32,)
                self.z_2 = tf.clip_by_value(self.z_2, -1., 1.)

            current_g = self.generator(tf.stop_gradient(self.z_2), self.batch_size, update_collection=None)
            D_lan = self.discriminator(current_g, self.batch_size, return_layers=False, update_collection="NO_OPS")
            _, k_gg_lan, k_ii_lan, k_gi_lan = self.mmd_loss(D_lan, D_images)

            g_loss_kl = - 2*tf.reduce_mean(tf.log(tf.reduce_mean(k_gi_lan, axis=-1) + 1e-10)) + tf.reduce_mean(tf.log(tf.reduce_sum(self.delete_diag(k_gg_lan), axis=-1)/(self.batch_size * (self.batch_size-1)) + 1e-10))
            d_loss_kl = tf.reduce_mean(tf.log(tf.reduce_mean(k_gi_lan, axis=-1) + 1e-10)) - tf.reduce_mean(tf.log(tf.reduce_mean(k_ii_lan, axis=-1) + 1e-10))


        # # when lam_1 = 2*lam_2, then it's simply (kde - 1/n)**2
        # square_g = 0.5*self.config.lam_1*tf.reduce_mean(tf.square(tf.reduce_sum(self.delete_diag(self.k_gg), axis=-1)/(self.batch_size-1)))

        square_g = self.config.lam_2*tf.reduce_mean(tf.square(tf.reduce_sum(self.delete_diag(self.k_gg), axis=-1)/(self.batch_size-1)))
        sim_g = self.config.lam_1*tf.reduce_mean(tf.reduce_sum(self.delete_diag(self.k_gg), axis=-1)/(self.batch_size-1))
        sim_gi = self.config.lam_3*tf.reduce_mean(self.k_gi)
        d_loss = square_g - sim_g + sim_gi
        

        if self.config.with_scaling:  # scaled objective
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                if self.config.use_gaussian_noise:
                    x_hat_data = tf.random_normal(self.images.get_shape().as_list(), mean=0.,
                                                  stddev=10., dtype=tf.float32, name='x_scaling')
                    x_hat = self.discriminator(x_hat_data, self.batch_size, update_collection="NO_OPS")
                else:
                    # Avoid rebuilding a new discriminator network subgraph
                    x_hat_data = self.images
                    x_hat = self.d_images
                norm2_jac = squared_norm_jacobian(x_hat, x_hat_data)
                norm2_jac = tf.reduce_mean(norm2_jac * scale_norm)

                # norm2_jac = self.delete_diag(self.cost_matrix(D_images, D_images, 'l2')/(self.cost_matrix(tf.reshape(self.images, (self.batch_size, -1)), tf.reshape(self.images, (self.batch_size, -1)), 'l2') + 1e-10))
                # norm2_jac = tf.reduce_sum(norm2_jac * scale_norm)/(self.batch_size * (self.batch_size-1))

                scale = 1. / (self.sc * norm2_jac + 1.)
                g_loss *= scale
                d_loss *= scale
                print('[*] Scaling added')
            else:
                if self.config.use_gaussian_noise:
                    x_hat_data = tf.random_normal(self.images.get_shape().as_list(), mean=0.,
                                                  stddev=10., dtype=tf.float32, name='x_scaling')
                    x_hat = self.discriminator(x_hat_data, self.batch_size, update_collection="NO_OPS")
                else:
                    # Avoid rebuilding a new discriminator network subgraph
                    x_hat_data = self.images
                    x_hat = self.d_images

                norm2_jac = squared_norm_jacobian(x_hat, x_hat_data)
                norm2_jac = tf.reduce_mean(norm2_jac)
                # norm2_jac = self.delete_diag(self.cost_matrix(D_images, D_images, 'l2')/(self.cost_matrix(tf.reshape(self.images, (self.batch_size, -1)), tf.reshape(self.images, (self.batch_size, -1)), 'l2') + 1e-10))
                # norm2_jac = tf.reduce_sum(norm2_jac)/(self.batch_size * (self.batch_size-1))

                scale = 1. / (self.sc * norm2_jac + 1.)
                g_loss *= scale
                d_loss *= scale
                print('[*] Scaling added')

        with tf.variable_scope('loss'):
            self.g_loss = g_loss
            self.d_loss = d_loss

            if self.config.train_lan_steps > 0:
                self.g_loss = self.g_loss * self.config.gamma_2 + self.config.gamma_1 * g_loss_kl
                self.d_loss = self.d_loss * self.config.gamma_2 + self.config.gamma_1 * d_loss_kl

            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.d_loss += self.config.ker_lam * noise_norm
                self.actual_noise_norm = noise_norm

            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.k_loss = self.d_loss

        self.optim_name = 'fp_loss'

        self.add_gradient_penalty(getattr(mmd, '_%s_kernel' % self.config.kernel), D_G, D_images)

        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
        print('[*] Loss set')

