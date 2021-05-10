from __future__ import division, print_function
import os
import sys
import time
import pprint

import numpy as np
from . import mmd
from .ops import safer_norm, tf, squared_norm_jacobian
from .architecture import get_networks
from .pipeline import get_pipeline
from utils import timer, scorer, misc
from tqdm import tqdm

# k_ratio = 0.005
# k_lam = 5.
k_variance = 0.


class MMD_GAN(object):
    def __init__(self, sess, config):

        self.k_ratio = config.k_ratio
        self.warm_up = 0.

        if config.learning_rate_D < 0:
            config.learning_rate_D = config.learning_rate
        """
        Args:
            sess: TensorFlow session
            config: The configuration; see main.py for entries
        """

        self.format = 'NCHW'
        self.timer = timer.Timer()
        self.dataset = config.dataset
        if config.architecture == 'dc128':
            config.output_size = 128
        elif config.architecture in ['dc64', 'dcgan64']:
            config.output_size = 64
        output_size = config.output_size

        self.sess = sess
        if config.real_batch_size == -1:
            config.real_batch_size = config.batch_size
        self.config = config
        self.is_grayscale = (config.c_dim == 1)
        self.batch_size = config.batch_size
        self.real_batch_size = config.real_batch_size
        self.sample_size = 64 if self.config.is_train else config.batch_size
        # self.sample_size = batch_size

        self.output_size = output_size
        self.data_dir = config.data_dir
        self.z_dim = self.config.z_dim

        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim
        self.dof_dim = self.config.dof_dim
        self.k_lam = config.ker_lam
        self.k_lam_2 = config.ker_lam_2
        self.c_dim = config.c_dim
        self.input_dim = self.output_size * self.output_size * self.c_dim

        discriminator_desc = '_dc'
        if self.config.learning_rate_D == self.config.learning_rate:
            lr = 'lr%.8f' % self.config.learning_rate
        else:
            lr = 'lr%.8fG%fD' % (self.config.learning_rate, self.config.learning_rate_D)
        arch = '%dx%d' % (self.config.gf_dim, self.config.df_dim)

        self.description = (
                    "%s%s_%s%s_%sd%d-%d-%d_%s_%s_%s_%.3g_%.3g_%.3g_%.3g_%i_%i_%.3g_%.3g_%.3g_%.3g_%.3g_%.3g_%.3g_%.3g_%.3g_%.3g_%.3g_%i" % (
                self.dataset, arch,
                self.config.architecture, discriminator_desc,
                self.config.model + '-' + self.config.kernel,
                self.config.dsteps,
                self.config.start_dsteps, self.config.gsteps, self.batch_size,
                self.output_size, lr, self.k_ratio, k_variance, self.k_lam, self.k_lam_2,
                int(self.config.with_scaling), int(self.config.with_sn), self.config.scaling_coeff,
                self.config.reg_ratio,
                self.config.eps, self.config.gamma_1, self.config.gamma_2, self.config.m_ratio, self.config.lam_1,
                self.config.lam_2, self.config.lam_3, self.config.lam_4, self.config.pico, self.config.msteps))
        if self.config.with_fp > 0.:
            self.description += '_with_fp{}'.format(self.config.with_fp)
        if self.config.dof_dim > 1:
            self.description += '_dof{}'.format(self.config.dof_dim)
        if self.config.batch_norm:
            self.description += '_bn'
        if self.config.train_lan_steps > 0:
            self.description += '_train_lan_step_{}'.format(self.config.train_lan_steps)
            self.description += '_lr_{}'.format(self.config.train_lan_step_lr)
        self.max_to_keep = 1
        self._ensure_dirs()
        self.with_labels = config.with_labels
        if self.with_labels:
            self.num_classes = 1000

        stdout = sys.stdout
        if self.config.log:
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            self.log_file = open(os.path.join(self.sample_dir, 'log.txt'), 'w', buffering=1)
            print('Execution start time: %s' % time.ctime())
            print('Log file: %s' % self.log_file)
            stdout = self.log_file
            sys.stdout = self.log_file
            sys.stderr = self.log_file
            print('Bullding the model')

        if config.compute_scores:
            self.scorer = scorer.Scorer(self.sess, self.dataset, config.MMD_lr_scheduler, stdout=stdout)
        print('Execution start time: %s' % time.ctime())

        pprint.PrettyPrinter().pprint(vars(self.config))
        print('Bullding the model')
        # if self.config.multi_gpu:
        #    self.build_model_multi_gpu()
        # else:
        self.build_model()
        self.initialized_for_sampling = config.is_train
        self.noise_norm_list = []
        print('Model ready')

    def _ensure_dirs(self, folders=['sample', 'log', 'checkpoint']):
        success = True
        if type(folders) == str:
            folders = [folders]
        for folder in folders:
            ff = folder + '_dir'

            self.__dict__[ff] = os.path.join(self.config.out_dir, vars(self.config)[ff],
                                             self.config.name, self.config.suffix +
                                             self.description)
            if not vars(self.config)[ff] == "":
                if not os.path.exists(self.__dict__[ff]):
                    os.makedirs(self.__dict__[ff])
            else:
                success = False
        return success

    def set_pipeline(self):
        Pipeline = get_pipeline(self.dataset, self.config.suffix)
        pipe = Pipeline(self.output_size, self.c_dim, self.real_batch_size,
                        os.path.join(self.data_dir, self.dataset), with_labels=self.with_labels, format=self.format,
                        timer=self.timer, sample_dir=self.sample_dir)
        if self.with_labels:
            self.image_batch, self.labels = pipe.connect()
        else:
            self.image_batch = pipe.connect()

        if self.format == 'NCHW':
            self.images_NHWC = tf.transpose(self.image_batch, [0, 2, 3, 1])
        else:
            self.images_NHWC = self.image_batch
        self.pipe = pipe

    def build_model(self):
        if self.config.multi_gpu:
            is_cpu_ps = True
            self.consolidation_device = '/cpu:2'
        else:
            is_cpu_ps = False
            self.consolidation_device = '/gpu:0'
            self.config.num_gpus = 1
        cpu_master_worker = '/cpu:1'
        cpu_data_processor = '/cpu:0'
        print('CPU ready')
        with tf.device(cpu_data_processor):
            self.set_pipeline()
            if self.with_labels:
                self.batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([self.image_batch, self.labels],
                                                                                 capacity=4 * self.config.num_gpus)

            else:
                self.batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([self.image_batch],
                                                                                 capacity=4 * self.config.num_gpus)
        print('Data ready')
        with tf.device(cpu_master_worker):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.global_d_step = tf.Variable(0, name="global_d_step", trainable=False)
            self.lr = tf.Variable(self.config.learning_rate, name='lr',
                                  trainable=False, dtype=tf.float32)
            self.lr_decay_op = self.lr.assign(tf.maximum(self.lr * self.config.decay_rate, 1.e-5))
            with tf.variable_scope('loss'):
                if self.config.is_train and (self.config.gradient_penalty > 0):
                    self.gp = tf.Variable(self.config.gradient_penalty,
                                          name='gradient_penalty',
                                          trainable=False, dtype=tf.float32)
                    self.gp_decay_op = self.gp.assign(self.gp * self.config.gp_decay_rate)
                if self.config.is_train and self.config.with_scaling:
                    self.sc = tf.Variable(self.config.scaling_coeff,
                                          name='scaling_coeff',
                                          trainable=False, dtype=tf.float32)
                    self.sc_decay_op = self.sc.assign(self.sc * self.config.sc_decay_rate)

            self.sample_z = tf.constant(np.random.uniform(-1, 1, size=(self.sample_size,
                                                                       self.z_dim)).astype(np.float32),
                                        dtype=tf.float32, name='sample_z')
            if self.with_labels:
                self.sample_y = tf.constant(np.random.choice(range(self.num_classes), size=(self.sample_size)),
                                            dtype=tf.int32, name='sample_y')
            Generator, Discriminator = get_networks(self.config.architecture)

        losses = []
        self.towers_g_grads = []
        self.towers_d_grads = []
        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            self.towers_k_grads = []

        self.update_ops = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.config.num_gpus):
                worker = '/gpu:%d' % i
                device_setter = misc._create_device_setter(is_cpu_ps, worker, self.config.num_gpus,
                                                           ps_device=self.consolidation_device)
                with tf.device(device_setter):

                    if self.with_labels:
                        images, labels = self.batch_queue.dequeue()
                        self.set_tower_loss('', images, Generator, Discriminator, labels=labels)

                    else:
                        images = self.batch_queue.dequeue()
                        self.set_tower_loss('', images, Generator, Discriminator)
                    tf.get_variable_scope().reuse_variables()
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        self.update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                        if self.config.is_train:
                            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                                losses.append([self.g_loss, self.d_loss, self.k_loss])
                            else:
                                losses.append([self.g_loss, self.d_loss])
                            if i == 0:
                                t_vars = tf.trainable_variables()
                                self.d_vars = [var for var in t_vars if 'd_' in var.name]
                                self.g_vars = [var for var in t_vars if 'g_' in var.name]
                                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                                    self.k_vars = [var for var in t_vars if 'k_' in var.name]
                            self.compute_grads()
                            self.towers_g_grads.append(self.g_gvs)
                            self.towers_d_grads.append(self.d_gvs)
                            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                                self.towers_k_grads.append(self.k_gvs)
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        if self.config.is_train:
            self.set_optimizer()

        block = min(8, int(np.sqrt(self.real_batch_size)), int(np.sqrt(self.batch_size)))

        summaries.append(tf.summary.image("train/input_image",
                                          self.imageRearrange(tf.clip_by_value(self.images, 0, 1), block)))
        summaries.append(tf.summary.image("train/gen_image",
                                          self.imageRearrange(tf.clip_by_value(self.G_NHWC, 0, 1), block)))
        # self.TrainSummary = tf.summary.merge(summaries)
        self.saver = tf.train.Saver(max_to_keep=10)
        print('[*] Model built.')

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
        Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def set_tower_loss(self, scope, images, Generator, Discriminator, update_collection=None, labels=None):
        self.images = images
        dbn = self.config.batch_norm & (self.config.gradient_penalty <= 0)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.,
                                   maxval=1., dtype=tf.float32, name='z')
        self.z_2 = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.,
                                     maxval=1., dtype=tf.float32, name='z_2')

        gen_kw = {
            'dim': self.gf_dim,
            'c_dim': self.c_dim,
            'output_size': self.output_size,
            'use_batch_norm': self.config.batch_norm,
            'format': self.format,
            'is_train': self.config.is_train,
        }
        disc_kw = {
            'dim': self.df_dim,
            'o_dim': self.dof_dim,
            'use_batch_norm': dbn,
            'with_sn': self.config.with_sn,
            'with_learnable_sn_scale': self.config.with_learnable_sn_scale,
            'format': self.format,
            'is_train': self.config.is_train,
            'scale': self.config.pico,
        }
        if self.with_labels:
            gen_kw['num_classes'] = disc_kw['num_classes'] = self.num_classes
        self.generator = Generator(**gen_kw)
        self.discriminator = Discriminator(**disc_kw)
        if self.config.model == 'w2flow_2d':
            self.discriminator_2 = Discriminator(**disc_kw, prefix='d_1_')

            # energy, langevin
        if self.config.lan_steps > 0:
            step_lr = self.config.lan_step_lr
            noise_std = tf.sqrt(step_lr * 2) * 0.01
            #             noise_std = 0.001
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            choice = self.config.lan_choice
            self.z_l = self.z
            d_i = self.discriminator(self.images, self.real_batch_size, return_layers=False, update_collection="NO_OPS")

            def delete_diag(matrix):
                return matrix - tf.matrix_diag(tf.matrix_diag_part(matrix))  # return matrix, while k_ii is 0

            for i in range(self.config.lan_steps):
                #                 d_i = self.discriminator(self.images,self.real_batch_size, return_layers=False, update_collection="NO_OPS")

                current_g = self.generator(self.z_l, self.batch_size, update_collection="NO_OPS")
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

                #                 if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                #                     kxx, kxy, _, _, _, _, _, _, _ = kernel(d_g, d_i,reg_ratio=self.config.reg_ratio)
                #                 else:
                #                     kxx, kxy, _, _, = kernel(d_g, d_i)

                if choice == 'kl':
                    # KL divergence
                    energy = -tf.log(tf.reduce_mean(kxy, axis=-1) + 1e-10) + tf.log(
                        tf.reduce_sum(delete_diag(kxx), axis=-1) / (self.config.batch_size - 1) + 1e-10)
                #                     energy = -tf.log(tf.reduce_mean(kxy, axis=-1)) + tf.log(tf.reduce_mean(kxx, axis=-1))
                elif choice == 'rkl':
                    # reverse KL divergence
                    energy = -(tf.reduce_mean(kxy, axis=-1) / (
                                tf.reduce_sum(delete_diag(kxx), axis=-1) / (self.config.batch_size - 1)) + 1e-10)
                #                     energy = -(tf.reduce_mean(kxy, axis=-1)/(tf.reduce_mean(kxx, axis=-1)))

                elif choice == 'js':
                    # Jensen-Shannon
                    energy = -0.5 * tf.log(tf.reduce_mean(kxy, axis=-1) + tf.reduce_sum(delete_diag(kxx), axis=-1) / (
                                self.config.batch_size - 1) + 1e-10) + 0.5 * tf.log(
                        2 * tf.reduce_sum(delete_diag(kxx), axis=-1) / (self.config.batch_size - 1) + 1e-10)
                #                     energy = -0.5*tf.log(tf.reduce_mean(kxy, axis=-1) + tf.reduce_mean(kxx, axis=-1)) + 0.5*tf.log(2*tf.reduce_mean(kxx, axis=-1))

                elif choice == 'sh':
                    # Squared Hellinger
                    energy = -tf.sqrt(tf.reduce_mean(kxy, axis=-1) / (
                                tf.reduce_sum(delete_diag(kxx), axis=-1) / (self.config.batch_size - 1)) + 1e-10)
                #                     energy = -tf.sqrt(tf.reduce_mean(kxy, axis=-1)/(tf.reduce_mean(kxx, axis=-1)))
                elif choice == 'lan':
                    energy = -tf.log(tf.reduce_mean(kxy, axis=-1) + 1e-10)
                else:
                    raise Exception("Undefined sampling method")

                z_grad = tf.gradients(energy, self.z_l)[0]
                if self.config.lan_clip > 0.:
                    z_grad = tf.clip_by_value(z_grad, -self.config.lan_clip, self.config.lan_clip)
                self.z_l = self.z_l - step_lr * z_grad
                self.z_l += tf.random_normal([self.batch_size, self.z_dim], mean=0., stddev=noise_std,
                                             dtype=tf.float32, )

                self.z_l = tf.clip_by_value(self.z_l, -1., 1.)

            self.G_lan = self.generator(self.z_l, self.batch_size, update_collection=update_collection)
            if self.format == 'NCHW':  # convert to NHWC format for sampling images
                self.G_lan = tf.transpose(self.G_lan, [0, 2, 3, 1])

       
        # tf.summary.histogram("z", self.z)
        if self.with_labels: 
            dist_y = tf.distributions.Categorical(probs=np.ones([self.num_classes]) / self.num_classes)
            self.y = dist_y.sample(sample_shape=[self.batch_size], name='y')
            self.G = self.generator(self.z, self.y, self.batch_size, update_collection=update_collection)
            self.sampler = self.generator(self.sample_z, self.sample_y, self.sample_size, update_collection="NO_OPS")
        else:
            self.G = self.generator(self.z, self.batch_size, update_collection=update_collection)
            self.sampler = self.generator(self.sample_z, self.sample_size, update_collection="NO_OPS")

        if self.format == 'NCHW':  # convert to NHWC format for sampling images
            self.G_NHWC = tf.transpose(self.G, [0, 2, 3, 1])
        else:
            self.G_NHWC = self.G

        if self.format == 'NCHW':  # convert to NHWC format for sampling images
            self.sampler = tf.transpose(self.sampler, [0, 2, 3, 1])

        if self.with_labels:
            self.d_images_layers = self.discriminator(self.images,
                                                      self.real_batch_size, return_layers=True,
                                                      update_collection=update_collection, y=labels)
            self.d_G_layers = self.discriminator(self.G, self.batch_size,
                                                 return_layers=True, update_collection="NO_OPS", y=self.y)
        else:
            self.d_images_layers = self.discriminator(self.images,
                                                      self.real_batch_size, return_layers=True,
                                                      update_collection=update_collection)
            self.d_G_layers = self.discriminator(self.G, self.batch_size,
                                                 return_layers=True, update_collection="NO_OPS")
        self.d_images = self.d_images_layers['hF']
        self.d_G = self.d_G_layers['hF']

        if self.config.is_train:
            self.set_loss(self.d_G, self.d_images)

    def set_loss(self, G, images):
        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            kerGI_0, kerGI_1, ker_GI_2, ker_GI_3, noise_norm, scale_norm, _, _, _ = kernel(G, images,
                                                                                           reg_ratio=self.config.reg_ratio)
        else:
            kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
            kerGI = kernel(G, images)

        with tf.variable_scope('loss'):
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.g_loss = mmd.mmd2([kerGI_0, kerGI_1, ker_GI_2, ker_GI_3]) + self.k_lam * noise_norm
                if not self.config.rep:
                    self.d_loss = -self.g_loss + (self.k_lam + self.k_lam_2) * noise_norm

                else:
                    self.d_loss = -mmd.mmd2([kerGI_0, kerGI_1, ker_GI_2, ker_GI_3], rep=0.5)
                self.optim_name = 'kernel_loss'
                self.k_loss = self.d_loss
                self.actual_noise_norm = tf.reduce_mean(scale_norm)
            else:
                self.g_loss = mmd.mmd2(kerGI)
                self.d_loss = -self.g_loss
                self.optim_name = 'kernel_loss'
            if self.config.with_fp > 0:
                kerGI_0, kerGI_1, ker_GI_2, ker_GI_3 = kerGI
                self.k_gg = kerGI_0
                self.k_ii = ker_GI_2
                self.k_gi = kerGI_1

#                 self.g_loss += self.config.with_fp * self.g_loss

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

        self.add_gradient_penalty(kernel, G, images)
        self.add_l2_penalty()

        print('[*] Loss set')

    def add_gradient_penalty(self, kernel, fake, real):
        bs = min([self.batch_size, self.real_batch_size])
        real, fake = real[:bs], fake[:bs]

        alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
        real_data = self.images[:bs]  # discirminator input level
        fake_data = self.G[:bs]  # discriminator input level
        x_hat_data = (1. - alpha) * real_data + alpha * fake_data
        x_hat = self.discriminator(x_hat_data, bs, update_collection="NO_OPS")
        Ekx = lambda yy: tf.reduce_mean(kernel(x_hat, yy, K_XY_only=True), axis=1)
        Ekxr, Ekxf = Ekx(real), Ekx(fake)
        witness = Ekxr - Ekxf
        gradients = tf.gradients(witness, [x_hat_data])[0]

        penalty = tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0))

        with tf.variable_scope('loss'):
            if self.config.gradient_penalty > 0:
                self.d_loss += penalty * self.gp
                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    self.k_loss = self.d_loss
                self.optim_name += '_(gp %.1f)' % self.config.gradient_penalty
                tf.summary.scalar('dx_penalty', penalty)
                print('[*] Gradient penalty added')
            tf.summary.scalar(self.optim_name + '_G', self.g_loss)
            tf.summary.scalar(self.optim_name + '_D', self.d_loss)
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                tf.summary.scalar(self.optim_name + '_K', self.k_loss)

    def add_l2_penalty(self):
        if self.config.L2_discriminator_penalty > 0:
            penalty = 0.0
            for _, layer in self.d_G_layers.items():
                penalty += tf.reduce_mean(tf.reshape(tf.square(layer), [self.batch_size, -1]), axis=1)
            for _, layer in self.d_images_layers.items():
                penalty += tf.reduce_mean(tf.reshape(tf.square(layer), [self.batch_size, -1]), axis=1)
            self.d_L2_penalty = self.config.L2_discriminator_penalty * tf.reduce_mean(penalty)
            self.d_loss += self.d_L2_penalty
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.k_loss = self.d_loss
            self.optim_name += ' (L2 dp %.6f)' % self.config.L2_discriminator_penalty
            self.optim_name = self.optim_name.replace(') (', ', ')
            tf.summary.scalar('L2_disc_penalty', self.d_L2_penalty)
            print('[*] L2 discriminator penalty added')

    def add_scaling_imp(self, scale_norm, noise_norm):
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
        norm_discriminator = tf.reduce_mean(tf.square(x_hat))

        scale = 1. / (self.sc * norm2_jac + 1.)

        unscaled_g_loss = self.g_loss
        with tf.variable_scope('loss'):
            if self.config.with_scaling:
                print('[*] Adding scaling variant: %s' % self.config.scaling_variant)
                self.apply_scaling_imp(scale, noise_norm)
                tf.summary.scalar(self.optim_name + '_non_scaled_G', unscaled_g_loss)
                tf.summary.scalar(self.optim_name + '_norm_grad_G', norm2_jac)
                tf.summary.scalar(self.optim_name + '_G', self.g_loss)
                tf.summary.scalar(self.optim_name + '_D', self.d_loss)
                tf.summary.scalar(self.optim_name + '_K', self.k_loss)
                tf.summary.scalar(self.optim_name + '_norm_D', norm_discriminator)
                print('[*] Scaling added')

    def add_scaling(self):
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
        norm_discriminator = tf.reduce_mean(tf.square(x_hat))

        if self.config.scaling_variant == 'grad':
            scale = 1. / (self.sc * norm2_jac + 1.)
        elif self.config.scaling_variant == 'value_and_grad':
            scale = 1. / (self.sc * (norm2_jac + norm_discriminator) + 1.)

        unscaled_g_loss = self.g_loss
        with tf.variable_scope('loss'):
            if self.config.with_scaling:

                print('[*] Adding scaling variant: %s' % self.config.scaling_variant)
                self.apply_scaling(scale)
                tf.summary.scalar(self.optim_name + '_non_scaled_G', unscaled_g_loss)
                tf.summary.scalar(self.optim_name + '_norm_grad_G', norm2_jac)
                tf.summary.scalar(self.optim_name + '_G', self.g_loss)
                tf.summary.scalar(self.optim_name + '_D', self.d_loss)
                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    tf.summary.scalar(self.optim_name + '_K', self.k_loss)
                tf.summary.scalar(self.optim_name + '_norm_D', norm_discriminator)
                print('[*] Scaling added')

    def set_optimizer(self):
        with tf.device(self.consolidation_device):
            with tf.control_dependencies(self.update_ops):
                self.g_gvs = self.average_gradients(self.towers_g_grads)
                self.d_gvs = self.average_gradients(self.towers_d_grads)
                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    self.k_gvs = self.average_gradients(self.towers_k_grads)
                self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.config.beta1, beta2=self.config.beta2)
                self.d_optim = tf.train.AdamOptimizer(self.lr * self.config.learning_rate_D / self.config.learning_rate,
                                                      beta1=self.config.beta1, beta2=self.config.beta2)
                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    self.k_optim = tf.train.AdamOptimizer(
                        self.k_ratio * self.lr * self.config.learning_rate_D / self.config.learning_rate,
                        beta1=self.config.beta1, beta2=self.config.beta2)
                self.apply_grads()

    def set_grads(self):
        with tf.variable_scope("G_grads"):
            self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.config.beta1, beta2=self.config.beta2)
            self.g_gvs = self.g_optim.compute_gradients(
                loss=self.g_loss,
                var_list=self.g_vars
            )

            if self.config.clip_grad:
                self.g_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.g_gvs]

            self.g_grads = self.g_optim.apply_gradients(
                self.g_gvs,
                global_step=self.global_step
            )

        with tf.variable_scope("D_grads"):
            self.d_optim = tf.train.AdamOptimizer(
                self.lr * self.config.learning_rate_D / self.config.learning_rate,
                beta1=self.config.beta1, beta2=self.config.beta2
            )
            self.d_gvs = self.d_optim.compute_gradients(
                loss=self.d_loss,
                var_list=self.d_vars
            )
            if self.config.clip_grad:
                self.d_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.d_gvs]
            self.d_grads = self.d_optim.apply_gradients(self.d_gvs)

        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            with tf.variable_scope("K_grads"):
                self.k_optim = tf.train.AdamOptimizer(
                    self.k_ratio * self.lr * self.config.learning_rate_D / self.config.learning_rate,
                    beta1=self.config.beta1, beta2=self.config.beta2
                )
                self.k_gvs = self.k_optim.compute_gradients(
                    loss=self.k_loss,
                    var_list=self.k_vars
                )
                if self.config.clip_grad:
                    self.k_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.k_gvs]
                self.k_grads = self.k_optim.apply_gradients(self.k_gvs)
        print('[*] Gradients set')

    def compute_grads(self):
        with tf.variable_scope("G_grads"):
            self.g_gvs = tf.gradients(self.g_loss, self.g_vars)
            self.g_gvs = zip(self.g_gvs, self.g_vars)
            if self.config.clip_grad:
                self.g_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.g_gvs]

        with tf.variable_scope("D_grads"):
            self.d_gvs = tf.gradients(self.d_loss, self.d_vars)
            self.d_gvs = zip(self.d_gvs, self.d_vars)
            if self.config.clip_grad:
                self.d_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.d_gvs]

        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            with tf.variable_scope("K_grads"):
                self.k_gvs = tf.gradients(self.k_loss, self.k_vars)
                self.k_gvs = zip(self.k_gvs, self.k_vars)
                if self.config.clip_grad:
                    self.k_gvs = [(tf.clip_by_norm(gg, 1.), vv) for gg, vv in self.k_gvs]
        print('[*] Gradients set')

    def apply_grads(self):
        with tf.variable_scope("G_grads"):
            if len(self.g_gvs):
                self.g_grads = self.g_optim.apply_gradients(
                    self.g_gvs,
                    global_step=self.global_step
                )
            else:
                self.d_grads = tf.no_op()
        with tf.variable_scope("D_grads"):
            if len(self.d_gvs):
                self.d_grads = self.d_optim.apply_gradients(
                    self.d_gvs,
                    global_step=self.global_d_step
                )
            else:
                self.d_grads = tf.no_op()
        if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
            with tf.variable_scope("K_grads"):
                if len(self.k_gvs):
                    self.k_grads = self.k_optim.apply_gradients(
                        self.k_gvs,
                        global_step=self.global_d_step
                    )
                else:
                    self.k_grads = tf.no_op()

    def set_counters(self, step):

        if (self.g_counter == 0) and (self.d_grads is not None):
            d_steps = self.config.dsteps
            # if step<10:
            if ((step % 500 == 0) or (step < 20)):
                d_steps = self.config.dsteps
            self.d_counter = (self.d_counter + 1) % (d_steps + 1)
        if self.d_counter == 0:
            self.g_counter = (self.g_counter + 1) % self.config.gsteps

    def set_summary(self, step, summary_str, g_loss, d_loss, write_summary, noise_norm=0.):
        if step % 100 == 0:
            try:
                #                 self.writer.add_summary(summary_str, step)
                self.err_counter = 0
            except Exception as e:
                print('Step %d summary exception. ' % step, e)
                self.err_counter += 1
        if write_summary:
            if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                self.timer(step,
                           "%s, G: %.8f, D: %.8f, Noise Norm: %.8f" % (self.optim_name, g_loss, d_loss, noise_norm))
            else:
                self.timer(step, "%s, G: %.8f, D: %.8f" % (self.optim_name, g_loss, d_loss))
            if self.config.L2_discriminator_penalty > 0:
                print(' ' * 22 + ('Discriminator L2 penalty: %.8f' % self.sess.run(self.d_L2_penalty)))

    def decay_ops(self):
        self.sess.run(self.lr_decay_op)
        if self.config.with_scaling:
            self.sess.run(self.sc_decay_op)

    def set_decay(self, step, is_init=False):
        if self.config.restart_lr:
            self.sess.run(self.lr.assign(self.config.learning_rate))
        if self.config.with_scaling and self.config.restart_sc:
            self.sess.run(self.sc.assign(self.config.scaling_coeff))
            print('current sc learning rate: %f' % self.sess.run(self.sc))

        print('current learning rate: %f' % self.sess.run(self.lr))

    def train_step(self, batch_images=None):
        step = self.sess.run(self.global_step)
        self.warm_up = min(max(0.5, step / 10000.), 1.)
        self.set_counters(step)
        write_summary = ((np.mod(step, 50) == 0) and (step < 1000)) \
                        or (np.mod(step, 5000) == 0) or (self.err_counter > 0)

        eval_ops = [self.g_gvs, self.d_gvs, self.g_loss, self.d_loss]
        #         print("step %d"%step)

        if self.config.is_demo:
            summary_str, g_grads, d_grads, g_loss, d_loss = self.sess.run(
                [self.TrainSummary] + eval_ops
            )
        else:
            if self.d_counter == 0:  # run the generator update
                if write_summary:
                    if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                        _, summary_str, g_grads, d_grads, g_loss, d_loss, noise_norm_0 = self.sess.run(
                            [self.g_grads, self.TrainSummary] + eval_ops + [self.actual_noise_norm]
                        )

                    else:
                        _, summary_str, g_grads, d_grads, g_loss, d_loss = self.sess.run(
                            [self.g_grads, self.TrainSummary] + eval_ops
                        )

                else:
                    if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                        _, g_grads, d_grads, g_loss, d_loss, noise_norm_0 = self.sess.run(
                            [self.g_grads] + eval_ops + [self.actual_noise_norm])
                    else:
                        _, g_grads, d_grads, g_loss, d_loss = self.sess.run([self.g_grads] + eval_ops)
            else:

                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    _, _, g_grads, d_grads, g_loss, d_loss, noise_norm_0 = self.sess.run(
                        [self.d_grads, self.k_grads] + eval_ops + [self.actual_noise_norm])

                else:
                    _, g_grads, d_grads, g_loss, d_loss = self.sess.run([self.d_grads] + eval_ops)

                # print("g loss: ",g_loss, ",  d loss:", d_loss)

            et = self.timer(step, "g step" if (self.d_counter == 0) else "d step", False)

        assert ~np.isnan(g_loss), et + "NaN g_loss, epoch: "
        assert ~np.isnan(d_loss), et + "NaN d_loss, epoch: "

        if self.d_counter == 0:
            if write_summary:
                if self.config.kernel == 'imp_1' or self.config.kernel == 'imp_2' or self.config.kernel == 'imp_3':
                    self.set_summary(step, summary_str, g_loss, d_loss, write_summary, noise_norm_0)
                    self.noise_norm_list.append(noise_norm_0)
                    np.savez(os.path.join(self.sample_dir, 'noise_norm_list.npz'),
                             norm=np.asarray(self.noise_norm_list))
                else:
                    self.set_summary(step, summary_str, g_loss, d_loss, write_summary)
            # self.set_decay(step)
            if self.config.compute_scores:
                self.scorer.compute(self, step)
        return g_loss, d_loss, step

    def train_init(self):
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        print('[*] Variables initialized.')
        self.TrainSummary = tf.summary.merge_all()
        self._ensure_dirs('log')
        #         self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.d_counter, self.g_counter, self.err_counter = 0, 0, 0

        if self.load_checkpoint():
            print(""" [*] Load SUCCESS, re-starting at epoch %d with learning
                  rate %.7f""" % (self.sess.run(self.global_step),
                                  self.sess.run(self.lr)))
        else:
            print(" [!] Load failed...")
            self.config.restart_lr = True

        step = self.sess.run(self.global_step)

        self.set_decay(step, is_init=True)

        print('[*] Model initialized for training')
        return step

    def train(self):
        step = self.train_init()
        self.pipe.start(self.sess)
        tf.train.start_queue_runners(sess=self.sess)
        pbar = tqdm(total=self.config.max_iteration + 1)
        while step <= self.config.max_iteration:
            prev_step = step
            g_loss, d_loss, step = self.train_step()
            if step > prev_step:
                pbar.update(1)
            self.save_checkpoint_and_samples(step)
            if self.config.save_layer_outputs:
                self.save_layers(step)
        self.pipe.stop()

    def save_checkpoint(self, step=None):
        if self._ensure_dirs('checkpoint'):
            if step is None:
                self.saver.save(self.sess,
                                os.path.join(self.checkpoint_dir, "best.model"))
            else:
                self.saver.save(self.sess,
                                os.path.join(self.checkpoint_dir, "MMDGAN.model"),
                                global_step=step)

    def load_checkpoint(self):
        print(" [*] Reading checkpoints...")
        if self.config.ckpt_name:
            ckpt_name = self.config.ckpt_name
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            #             print(self.checkpoint_dir, ckpt, ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False

        print('generator path: ', self.checkpoint_dir)
        self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))

        if self.config.pretrained_kernel == '':
            return True

        else:
            print('discriminator (kernel) path: ', self.config.pretrained_kernel)
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'd_' in var.name]
            k_vars = [var for var in t_vars if 'k_' in var.name]
            saver_d = tf.train.Saver(d_vars + k_vars)

            ckpt_d = tf.train.get_checkpoint_state(self.config.pretrained_kernel)
            if ckpt_d and ckpt_d.model_checkpoint_path:
                print('reading name')
                ckpt_d_name = os.path.basename(ckpt_d.model_checkpoint_path)
                print(ckpt_d_name)
            else:
                print('fail in reading pretrained discriminator (kernel)')
                return False
            saver_d.restore(self.sess, os.path.join(self.config.pretrained_kernel, ckpt_d_name))
            return True

    def save_checkpoint_and_samples(self, step, freq=1000):
        checkpoint_freq = 2000
        sample_freq = 1000
        if (np.mod(step, checkpoint_freq) == 0) and (self.d_counter == 0):
            self.save_checkpoint(step)
        if (np.mod(step, sample_freq) == 0) and (self.d_counter == 0):
            samples = self.sess.run(self.sampler)
            self._ensure_dirs('sample')
            p = os.path.join(self.sample_dir, 'train_{:02d}.png'.format(step))
            misc.save_images(samples[:64, :, :, :], [8, 8], p)

    def save_layers(self, step, freq=1000, n=256, layers=[-1, -2]):
        c = self.config.save_layer_outputs
        valid = list(freq * np.arange(self.config.max_iteration / freq + 1))
        if c > 1:
            valid += [int(k) for k in c ** np.arange(np.log(freq) / np.log(c))]
        if (step in valid) and (self.d_counter == 0):
            if not (layers == 'all'):
                keys = [sorted(list(self.d_G_layers))[i] for i in layers]
            fake = [(key + '_fake', self.d_G_layers[key]) for key in keys]
            real = [(key + '_real', self.d_images_layers[key]) for key in keys]

            values = self._evaluate_tensors(dict(real + fake), n=n)
            path = os.path.join(self.sample_dir, 'layer_outputs_%d.npz' % step)
            np.savez(path, **values)

    def imageRearrange(self, image, block=4):
        image = tf.slice(image, [0, 0, 0, 0], [block * block, -1, -1, -1])
        x1 = tf.batch_to_space(image, [[0, 0], [0, 0]], block)
        image_r = tf.reshape(
            tf.transpose(
                tf.reshape(
                    x1,
                    [self.output_size, block, self.output_size, block, self.c_dim]
                ),
                [1, 0, 3, 2, 4]
            ),
            [1, self.output_size * block, self.output_size * block, self.c_dim]
        )
        return image_r

    def _evaluate_tensors(self, variable_dict, n=None):
        if n is None:
            n = self.batch_size
        values = dict([(key, []) for key in variable_dict.keys()])
        sampled = 0
        while sampled < n:
            vv = self.sess.run(variable_dict)
            for key, val in vv.items():
                values[key].append(val)
            sampled += list(vv.items())[0][1].shape[0]
        for key, val in values.items():
            values[key] = np.concatenate(val, axis=0)[:n]
        return values

    def get_samples(self, n=None, save=True, layers=[]):
        if not (self.initialized_for_sampling or self.config.is_train):
            print('[*] Loading from ' + self.checkpoint_dir + '...')
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())
            if self.load_checkpoint():
                print(" [*] Load SUCCESS, model trained up to epoch %d" %
                      self.sess.run(self.global_step))
            else:
                print(" [!] Load failed...")
                return

        if len(layers) > 0:
            outputs = dict([(key + '_features', val) for key, val in self.d_G_layers.items()])
            if not (layers == 'all'):
                keys = [sorted(list(outputs.keys()))[i] for i in layers]
                outputs = dict([(key, outputs[key]) for key in keys])
        else:
            outputs = {}
        if self.config.lan_steps > 0:

            outputs['samples'] = self.G_lan
        else:
            outputs['samples'] = self.G_NHWC

        values = self._evaluate_tensors(outputs, n=n)

        if not save:
            if len(layers) > 0:
                return values
            return values['samples']

        for key, val in values.items():
            file = os.path.join(self.sample_dir, '%s.npy' % key)
            np.save(file, val, allow_pickle=False)
            print(" [*] %d %s saved in '%s'" % (n, key, file))
