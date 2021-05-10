from .model import MMD_GAN, tf
from . import mmd
from .ops import safer_norm, tf, squared_norm_jacobian
from .ops import jacob
slim = tf.contrib.slim
import math
class W2flow(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        self.config = config
        self.batch_size = self.config.batch_size
        super(W2flow, self).__init__(sess, config, **kwargs)

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

    def compute_loss(self, x, y, mode, dis='1', kde=False):  # X and Y are batch of samples/transferred samples here, X is the real, Y is the fake
        if kde:  # we compute the marginal probability according to riemannian volume using heat kernel...
            if dis == '1':
                # mu = self.compute_kde(self.k_gg)   # 1st ways to set the KDE, in this case, \nu=tf.stop_gradient(\mu)
                # nu = tf.stop_gradient(mu)  # 1st ways to set the KDE, in this case, \nu=tf.stop_gradient(\mu)

                mu = self.compute_kde(self.k_gg)
                mu /= tf.stop_gradient(mu)*self.batch_size    # 2nd way to set the KDE, in this case, \nu=1/batch_size
                nu = tf.ones(self.batch_size) / self.batch_size

                # mu /= tf.stop_gradient(tf.reduce_sum(mu))    # normalize it ?
                # nu /= tf.stop_gradient(nu)
                self.mu_1 = mu
            else:
                # mu = self.compute_kde(self.k_ii)   # 1st ways to set the KDE, in this case, \nu=tf.stop_gradient(\mu)
                # nu = tf.stop_gradient(mu)  # 1st ways to set the KDE, in this case, \nu=tf.stop_gradient(\mu)

                mu = self.compute_kde(self.k_ii)
                mu /= tf.stop_gradient(mu)*self.batch_size    # 2nd way to set the KDE, in this case, \nu=1/batch_size
                nu = tf.ones(self.batch_size) / self.batch_size

                # mu /= tf.stop_gradient(tf.reduce_sum(mu))    # normalize it ?
                # nu /= tf.stop_gradient(nu)
                self.mu_2 = mu
        else:
            mu = tf.ones(self.batch_size)/self.batch_size  # shape (batch_size)
            nu = tf.ones(self.batch_size)/self.batch_size  # shape (batch_size)
        epsilon = self.config.eps
        u = mu*0.
        v = nu*0.
        self.c = self.cost_matrix(x, tf.stop_gradient(y), mode)
        for i in range(20):
            u += (tf.log(mu + 1e-7) - tf.log(tf.reduce_sum(tf.exp(self.M(u, v, epsilon)), axis=-1) + 1e-7))
            v += (tf.log(nu + 1e-7) - tf.log(tf.reduce_sum(tf.exp(tf.transpose(self.M(u, v, epsilon))), axis=-1) + 1e-7))
        pi = tf.exp(self.M(u, v, epsilon))  # pi is the transportation plan, i.e. the coupling
        pi /= tf.stop_gradient(tf.reduce_sum(pi) + 1e-7)  # the sinkhorn algorithm produce a unique solution UP TO A CONSTANT, we make it a joint probability
        cost = tf.reduce_sum(pi*self.c)
        return cost, pi

    def compute_sinkhorn_loss(self, G, images, t=True, mode='l2', dis='1', kde=False):
        if t:  # which means we compute the w2 distance between current and previous time step
            if dis == '1':
                # sinkhorn_loss, coupling_matrix = self.compute_loss(G, G, mode, dis=dis, kde=kde)  # projected distance (in feature space)?
                sinkhorn_loss, coupling_matrix = self.compute_loss(tf.reshape(self.G, (self.batch_size, -1)), tf.reshape(self.G, (self.batch_size, -1)), mode, dis=dis, kde=kde)  # distance (in input space)?

            else:
                # sinkhorn_loss, coupling_matrix = self.compute_loss(images, images, mode, dis=dis, kde=kde)  # projected distance (in feature space)?
                sinkhorn_loss, coupling_matrix = self.compute_loss(tf.reshape(self.images, (self.batch_size, -1)), tf.reshape(self.images, (self.batch_size, -1)), mode, dis=dis, kde=kde)  # distance (in input space)?

        else:
            sinkhorn_loss, coupling_matrix = self.compute_loss(G, images, mode, dis=dis, kde=kde)
        return sinkhorn_loss, coupling_matrix

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


        g_loss = self.mmd_1
        diffusion_loss = - tf.reduce_sum(self.delete_diag(self.k_gg))/(self.batch_size * (self.batch_size-1)) \
                          + tf.reduce_sum(self.delete_diag(self.config.lam_3*self.k_ii_sin - self.k_ii +
                                                            self.config.lam_4*(self.cost_matrix(D_images, D_images, 'l2'))
                                                           )
                                          )/(self.batch_size * (self.batch_size-1)) \
                          + tf.reduce_mean(self.config.lam_1*self.k_gi_sin - self.k_gi +
                                                self.config.lam_2 *(self.cost_matrix(D_G, D_images, 'l2'))
                                           )
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
                diffusion_loss *= scale
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
                diffusion_loss *= scale
                print('[*] Scaling added')

        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            diffusion_loss += self.config.ker_lam * noise_norm
            self.actual_noise_norm = noise_norm
        w2_loss, _ = self.compute_sinkhorn_loss(D_G, D_images, t=True, mode='l2', dis='1', kde=True)
        # energy_loss = tf.reduce_sum(tf.log(self.mu_1)*self.mu_1)  # this is H(\mu) = \sum_{i=1}^n \mu(x_i) log(\mu(x_i))
        # energy_loss = tf.reduce_sum(tf.log(k_gg + 1e-7)*self.mu_1)/self.batch_size  # this is H(\mu) = 1/n \sum_{j=1}^n\sum_{i=1}^n \mu(x_i) log(k(x_i, x_j))
        energy_loss = tf.reduce_sum(self.delete_diag(tf.log(k_gg + 1e-7))*self.mu_1)/(self.batch_size-1)  # this is equivalent to the above line, we delete diagonal here, which has no impact because diagonals are k(x,x)=1, log k(x,x)=0

        with tf.variable_scope('loss'):
            self.g_loss = g_loss
            self.d_loss = diffusion_loss + self.config.gamma_1*energy_loss + self.config.gamma_2*w2_loss
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.k_loss = self.d_loss
        self.optim_name = 'w2flow_loss'

        self.add_gradient_penalty(getattr(mmd, '_%s_kernel' % self.config.kernel), D_G, D_images)

        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
        print('[*] Loss set')

