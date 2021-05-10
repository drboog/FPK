#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import scipy
import sys
import numpy as np
from core import mmd
import compute_scores as cs
import skimage
from utils import misc

class Scorer(object):
    def __init__(self, sess, dataset, lr_scheduler=True, stdout=sys.stdout):
        self.stdout = stdout
        self.dataset = dataset
        if dataset == 'mnist':
            self.model = cs.LeNet(sess)
            self.size = 100000
            self.frequency = 500

        elif dataset == 'celeba':
            self.model = cs.Inception(sess)
            self.size = 50000
            self.frequency = 2000

        elif dataset == 'cifar10':
            self.model = cs.Inception(sess)
            self.size = 50000
            self.frequency = 2000

        elif dataset == 'stl10':
            self.model = cs.Inception(sess)
            self.size = 100000
            self.frequency = 2000


        self.output = []

        if lr_scheduler:
            self.three_sample = []
            self.three_sample_chances = 0
        self.lr_scheduler = lr_scheduler
        
    def get_info_of_imgs(self, imgs):
        print('Image group shape is: ', np.shape(imgs), ', max value is: ', np.max(imgs), ' min value is: ', np.min(imgs))
        
    def set_train_codes(self, gan):
        suffix = '' if (gan.output_size <= 32) else ('-%d' % gan.output_size)
        path = os.path.join(gan.data_dir, '%s-codes%s.npy' % (self.dataset, suffix))

        if os.path.exists(path):
            self.train_codes = np.load(path)
            print('[*] Train codes loaded. ')
            return
        print('[!] Codes not found. Featurizing...')
        ims = []
        while len(ims) < self.size // gan.batch_size:
            ims.append(gan.sess.run(gan.images_NHWC))
        ims = np.concatenate(ims, axis=0)[:self.size]
        self.get_info_of_imgs(ims)
        if self.dataset == 'mnist':  # LeNet model takes [-.5, .5] pics
            ims -= .5
            self.get_info_of_imgs(ims)
            if (ims.max() > .5) or (ims.min() < -.5):
                print('WARNING! LeNet min/max violated: min = %f, max = %f. Clipping values.' % (ims.min(), ims.max()))
                ims = ims.clip(-.5, .5)
        else:
            ims *= 255# skimage.img_as_ubyte(ims) # transfer into [0, 255]
            self.get_info_of_imgs(ims)
            if (ims.max() > 255.) or (ims.min() < .0):
                print('WARNING! Inception min/max violated: min = %f, max = %f. Clipping values.' % (ims.min(), ims.max()))
                ims = ims.clip(0., 255.)

        _, self.train_codes = cs.featurize(ims, self.model, get_preds=True,
                                           get_codes=True, output=self.stdout)
        np.save(path, self.train_codes)
        print('[*] %d train images featurized and saved in <%s>' % (self.size, path))

    def compute(self, gan, step):
        if step % gan.config.MMD_sdlr_freq != 0:
            return

        if not hasattr(self, 'train_codes'):
            print('[ ] Getting train codes...')
            self.set_train_codes(gan)

        tt = time.time()
        gan.timer(step, "Scoring start")
        output = {}
        images4score = gan.get_samples(n=self.size, save=False)
        self.get_info_of_imgs(images4score)
        if self.dataset == 'mnist':  # LeNet model takes [-.5, .5] pics
            images4score -= .5
            self.get_info_of_imgs(images4score)
            if (images4score.max() > .5) or (images4score.min() < -.5):
                print('WARNING! LeNet min/max violated: min = %f, max = %f. Clipping values.' % (images4score.min(), images4score.max()))
                images4score = images4score.clip(-.5, .5)
        else:  # Inception model takes [0 , 255] pics
            images4score *= 255#skimage.img_as_ubyte(images4score) # transfer into [0, 255]
            self.get_info_of_imgs(images4score)
            if (images4score.max() > 255.) or (images4score.min() < .0):
                print('WARNING! Inception min/max violated: min = %f, max = %f. Clipping values.' % (images4score.min(), images4score.max()))
                images4score = images4score.clip(0., 255.)
        print(np.shape(images4score), np.max(images4score), np.min(images4score))
        preds, codes = cs.featurize(images4score, self.model, get_preds=True,
                                    get_codes=True, output=self.stdout)
        print(np.max(preds), np.min(preds), np.shape(preds))
        preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
        gan.timer(step, "featurizing finished")

        output['inception'] = scores = cs.inception_score(preds)
        print('IS', np.mean(scores), np.std(scores))
        gan.timer(step, "Inception mean (std): %f (%f)" % (np.mean(scores), np.std(scores)))
        output['fid'] = scores = cs.fid_score(
            codes,
            self.train_codes,
            output=self.stdout,
            split_method='bootstrap',
            splits=3)
        gan.timer(step, "FID mean (std): %f (%f)" % (np.mean(scores), np.std(scores)))
        print('FID: ', np.mean(scores), np.std(scores))
        
        samples = gan.sess.run(gan.sampler)
        gan._ensure_dirs('sample')
        p = os.path.join(gan.sample_dir, 'train_{:02d}.png'.format(step))
        misc.save_images(samples[:64, :, :, :], [8, 8], p)
            
        ret = cs.polynomial_mmd_averages(
            codes,
            self.train_codes,
            output=self.stdout,
            n_subsets=10,
            subset_size=1000,
            ret_var=False)
        output['mmd2'] = mmd2s = ret
        gan.timer(step, "KID mean (std): %f (%f)" % (mmd2s.mean(), mmd2s.std()))

        if len(self.output) > 0:
            if np.min([sc['mmd2'].mean() for sc in self.output]) > output['mmd2'].mean():
                print('Saving BEST model (so far)')
                gan.save_checkpoint()
        self.output.append(output)

        if self.lr_scheduler:
            n = gan.config.MMD_sdlr_past_sample
            nc = gan.config.MMD_sdlr_num_test
            bs = 2048
            new_Y = codes[:bs]
            X = self.train_codes[:bs]
            print('3-sample stats so far: %d' % len(self.three_sample))
            if len(self.three_sample) >= n:
                saved_Z = self.three_sample[0]
                mmd2_diff, test_stat, Y_related_sums = \
                    mmd.np_diff_polynomial_mmd2_and_ratio_with_saving(X, new_Y, saved_Z)
                p_val = scipy.stats.norm.cdf(test_stat)
                gan.timer(step, "3-sample test stat = %.1f" % test_stat)
                gan.timer(step, "3-sample p-value = %.1f" % p_val)
                if p_val > .1:
                    self.three_sample_chances += 1
                    if self.three_sample_chances >= nc:
                        # no confidence that new Y sample is closer to X than old Z is
                        gan.decay_ops()
                        print('No improvement in last %d tests. Decreasing learning rate to %f' %
                              (nc, gan.sess.run(gan.lr)))
                        if gan.config.with_scaling:
                            print(' Decreasing scaling amplitude to %f' %
                                  gan.sess.run(gan.sc))
                        self.three_sample = (self.three_sample + [Y_related_sums])[-nc:]  # reset memorized sums
                        self.three_sample_chances = 0
                    else:
                        print('No improvement in last %d test(s). Keeping learning rate at %f' %
                              (self.three_sample_chances, gan.sess.run(gan.lr)))
                        if gan.config.with_scaling:
                            print(' Keeping scaling amplitude to %f' %
                                  gan.sess.run(gan.sc))
                else:
                    # we're confident that new_Y is better than old_Z is
                    print('Keeping learning rate at %f' % gan.sess.run(gan.lr))
                    if gan.config.with_scaling:
                        print(' Keeping scaling amplitude to %f' %
                              gan.sess.run(gan.sc))
                    self.three_sample = self.three_sample[1:] + [Y_related_sums]
                    self.three_sample_chances = 0
            else:  # add new sums to memory
                self.three_sample.append(
                    mmd.np_diff_polynomial_mmd2_and_ratio_with_saving(X, new_Y, None)
                )
                gan.timer(step, "computing stats for 3-sample test finished")
                print('current learning rate: %f' % gan.sess.run(gan.lr))
                if gan.config.with_scaling:
                        print(' current scaling amplitude to %f' %
                              gan.sess.run(gan.sc))
        filepath = os.path.join(gan.sample_dir, 'score%d.npz' % step)

        output['lr'] = np.array([gan.sess.run(gan.lr)])
        if gan.config.with_scaling:
            output['sc'] = np.array([gan.sess.run(gan.sc)])

        np.savez(filepath, **output)
        gan.timer(step, "Scoring end, total time = %.1f s" % (time.time() - tt))
