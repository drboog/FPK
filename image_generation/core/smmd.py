from .model import MMD_GAN
from . import mmd
from .ops import tf

class SMMD(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        super(SMMD, self).__init__(sess, config, **kwargs)

    def set_loss(self, G, images):
        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            kerGI_0, kerGI_1, ker_GI_2, ker_GI_3, noise_norm, scale_norm, _, _, _ = kernel(G, images, reg_ratio=self.config.reg_ratio)
        else:
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            kerGI = kernel(G, images)

#         kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
#         kerGI = kernel(G, images)
        
        with tf.variable_scope('loss'):
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.g_loss = mmd.mmd2([kerGI_0, kerGI_1, ker_GI_2, ker_GI_3])
                self.d_loss = -self.g_loss
                self.optim_name = 'kernel_loss'
                self.k_loss = self.d_loss# + self.k_lam*noise_norm
                self.actual_noise_norm = tf.reduce_mean(scale_norm)   
            else:
                self.g_loss = mmd.mmd2(kerGI)
                self.d_loss = -self.g_loss
                self.optim_name = 'kernel_loss'
                
#         with tf.variable_scope('loss'):
#             self.g_loss = mmd.mmd2(kerGI) 
#             self.d_loss = -self.g_loss
#             self.optim_name = 'kernel_loss'
        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            self.add_scaling_imp(scale_norm, noise_norm)
        else:
            self.add_scaling()
        print('[*] Loss set')

    def apply_scaling(self, scale):
        self.g_loss = self.g_loss*scale
        self.d_loss = -self.g_loss

    def apply_scaling_imp(self, scale, noise_norm):
        self.g_loss = self.g_loss*scale + self.k_lam*noise_norm
        self.d_loss = -self.g_loss + (self.k_lam + self.k_lam_2)*noise_norm
        self.k_loss = self.d_loss

class SWGAN(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        config.dof_dim = 1
        super(SWGAN, self).__init__(sess, config, **kwargs)

    def set_loss(self, G, images):

        with tf.variable_scope('loss'):
            self.d_loss = tf.reduce_mean(G) - tf.reduce_mean(images)
            self.g_loss = -self.d_loss
            self.optim_name = 'swgan_loss'
        self.add_scaling()
        print('[*] Loss set')

    def apply_scaling(self, scale):
        self.g_loss = self.g_loss*tf.sqrt(scale)
        self.d_loss = -self.g_loss
