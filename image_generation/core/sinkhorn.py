from .model import MMD_GAN, tf
from . import mmd


class Sinkhorn(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        self.config = config
        super(Sinkhorn, self).__init__(sess, config, **kwargs)

    def delete_diag(self, matrix):
        return matrix - tf.matrix_diag(tf.matrix_diag_part(matrix))  # return matrix, while k_ii is 0
    
    def cost_matrix(self, x, y):  # compute the cost matrix (L2 distances)
        x_expand = tf.expand_dims(x, axis=-2)
        y_expand = tf.expand_dims(y, axis=-3)
        c = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)  # sum over the dimensions
        return c

    def M(self, u, v, eps):
        return (-self.c + tf.expand_dims(u, -2) + tf.expand_dims(v,
                                                                 -1)) / eps  # the return shape is (batch_size, batch_size)
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

    def compute_loss(self, x, y):  # X and Y are batch of samples/transferred samples here, X is the real, Y is the fake
        self.c = self.cost_matrix(x, y)
        mu = tf.ones(self.batch_size) / self.batch_size  # shape (batch_size)
        nu = tf.ones(self.batch_size) / self.batch_size  # shape (batch_size)
        threshold = 10 ** (-1)  # threshold to stop the iteration
        epsilon = self.config.eps
        u = mu * 0.
        v = nu * 0.
        err = 0.  # some initialization
        for i in range(20):
            u1 = u  # used to check the error later
            u += (tf.log(mu) - tf.log(tf.reduce_sum(tf.exp(self.M(u, v, epsilon)), axis=-1) + 1e-6))
            v += (tf.log(nu) - tf.log(tf.reduce_sum(tf.exp(tf.transpose(self.M(u, v, epsilon))), axis=-1) + 1e-6))
            # err = tf.reduce_sum(tf.abs(u - u1))
        pi = tf.exp(self.M(u, v, epsilon))  # pi is the transportation plan, i.e. the coupling
        cost = tf.reduce_sum(pi * self.c)
        return cost, tf.exp(-self.c / epsilon)

    
    def delete_diag(self, matrix):
        return matrix - tf.matrix_diag(tf.matrix_diag_part(matrix))  # return matrix, while k_ii is 0
    
    
    def set_loss(self, G, images):
        D_G = G  # self.discriminator(G, self.batch_size)
        D_images = images  # self.discriminator(images, self.batch_size)
        sinkhorn_loss, kernel_matrix = self.compute_loss(D_G, D_images)
        sinkhorn_loss_1, kernel_matrix_1 = self.compute_loss(D_G, D_G)
        sinkhorn_loss_2, kernel_matrix_2 = self.compute_loss(D_images, D_images)
        with tf.variable_scope('loss'):
            self.g_loss = (2 * sinkhorn_loss - sinkhorn_loss_1 - sinkhorn_loss_2)
            self.d_loss = -self.g_loss
            if self.config.with_fp > 0:
                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    mmd_1, k_gg, k_ii, k_gi, noise_norm, scale_norm, k_gg_sin, k_ii_sin, k_gi_sin = self.mmd_loss(D_G,
                                                                                                                  D_images)
                    self.k_gg_sin = k_gg_sin
                    self.k_ii_sin = k_ii_sin
                    self.k_gi_sin = k_gi_sin
                else:
                    mmd_1, k_gg, k_ii, k_gi = self.mmd_loss(D_G,
                                                            D_images)  # k_gg means generated images kernel matrix ...
                    self.k_gg_sin = k_gg
                    self.k_ii_sin = k_ii
                    self.k_gi_sin = k_gi
                self.k_gg = k_gg
                self.k_ii = k_ii
                self.k_gi = k_gi
                self.mmd_1 = mmd_1

#                 self.g_loss += self.config.with_fp * self.mmd_1

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
        self.optim_name = 'sinkhorn_loss'
        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
        print('[*] Loss set')
