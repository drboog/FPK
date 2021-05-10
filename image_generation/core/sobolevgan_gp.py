from .model import MMD_GAN, tf


class SobolevGAN(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        config.dof_dim = 1
        super(SobolevGAN, self).__init__(sess, config, **kwargs)

    def set_loss(self, G, images):
        real_data = self.images
        fake_data = self.G

        d_real = self.d_images
        d_fake = self.d_G

        gradients_real = tf.gradients(d_real, [real_data])[0]
        gradients_fake = tf.gradients(d_fake, [fake_data])[0]

        e_norm_grad_real = tf.reduce_mean(tf.reduce_sum(tf.square(gradients_real), reduction_indices=[1, 2, 3]))
        e_norm_grad_fake = tf.reduce_mean(tf.reduce_sum(tf.square(gradients_fake), reduction_indices=[1, 2, 3]))

        self.sobolev_constraint = 0.5*e_norm_grad_real + 0.5*e_norm_grad_fake - 1.

        self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                  initializer=self.config.gradient_penalty)

        self.lambda_sobolev = tf.Variable(0., name='lambda_sobolev', dtype=tf.float32, trainable=False)

        self.d_loss = tf.reduce_mean(G) - tf.reduce_mean(images)
        self.g_loss = -tf.reduce_mean(G)

        weights_decay = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        linear_decay = tf.reduce_sum(tf.square(weights_decay[-1])) + tf.reduce_sum(tf.square(weights_decay[-2]))
        weights_decay = weights_decay[:-2]
        conv_decay = tf.reduce_sum(tf.stack([tf.reduce_sum(tf.square(w)) for w in weights_decay]))
        decay_linear_param = 0.001
        decay_conv_param = 0.000001

        if self.config.sobolev_gan:
            self.d_loss += self.lambda_sobolev * self.sobolev_constraint + 0.5*self.gp*self.sobolev_constraint**2
            self.d_loss += decay_conv_param * conv_decay + decay_linear_param * linear_decay
            self.lambda_sobolev_op = self.lambda_sobolev.assign(self.lambda_sobolev + self.sobolev_constraint * self.gp)
        self.optim_name = 'sobolevgan%d_loss' % int(self.config.gradient_penalty)

        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
