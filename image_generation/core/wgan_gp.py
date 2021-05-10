from .model import MMD_GAN, tf
from . import mmd


class WGAN_GP(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        config.dof_dim = 1
        super(WGAN_GP, self).__init__(sess, config, **kwargs)

    def delete_diag(self, matrix):
        return matrix - tf.matrix_diag(tf.matrix_diag_part(matrix))  # return matrix, while k_ii is 0
    
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

    def set_loss(self, G, images):
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1])
        real_data = self.images
        fake_data = self.G
        differences = fake_data - real_data
        interpolates0 = real_data + (alpha * differences)
        interpolates = self.discriminator(interpolates0, self.batch_size)

        gradients = tf.gradients(interpolates, [interpolates0])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                  initializer=self.config.gradient_penalty)

        self.d_loss = tf.reduce_mean(G) - tf.reduce_mean(images)
        self.g_loss = -tf.reduce_mean(G)
        if self.config.gradient_penalty > 0:
            self.d_loss += self.gp * gradient_penalty

        if self.config.with_fp > 0:
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                mmd_1, k_gg, k_ii, k_gi, noise_norm, scale_norm, k_gg_sin, k_ii_sin, k_gi_sin = self.mmd_loss(G, images)
                self.k_gg_sin = k_gg_sin
                self.k_ii_sin = k_ii_sin
                self.k_gi_sin = k_gi_sin
            else:
                mmd_1, k_gg, k_ii, k_gi = self.mmd_loss(G, images)  # k_gg means generated images kernel matrix ...
                self.k_gg_sin = k_gg
                self.k_ii_sin = k_ii
                self.k_gi_sin = k_gi
            self.k_gg = k_gg
            self.k_ii = k_ii
            self.k_gi = k_gi
            self.mmd_1 = mmd_1

#             self.g_loss += self.config.with_fp * self.mmd_1

            # when lam_1 = 2*lam_2, then it's simply (kde - 1/n)**2
            #         square_g = 0.5*self.config.lam_1*tf.reduce_mean(tf.square(tf.reduce_sum(self.delete_diag(self.k_gg), axis=-1)/(self.batch_size-1)))

            square_g = self.config.lam_2 * tf.reduce_mean(
                tf.square(tf.reduce_sum(self.delete_diag(self.k_gg), axis=-1) / (self.batch_size - 1)))
            sim_g = self.config.lam_1 * tf.reduce_mean(
                tf.reduce_sum(self.delete_diag(self.k_gg), axis=-1) / (self.batch_size - 1))
            sim_gi = self.config.lam_3 * tf.reduce_mean(self.k_gi)
            self.d_loss += self.config.with_fp * (square_g - sim_g + sim_gi)
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.d_loss += self.config.with_fp * (self.config.ker_lam * noise_norm)
                self.k_loss = self.d_loss

        self.optim_name = 'wgan_gp%d_loss' % int(self.config.gradient_penalty)

        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
