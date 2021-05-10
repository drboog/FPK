from .model import MMD_GAN, tf
from tensorflow import expand_dims as E


class KernelSobolevMMD_GAN(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        super(KernelSobolevMMD_GAN, self).__init__(sess, config, **kwargs)

    def set_loss(self, G, images):
        input_img_shape = self.images.get_shape()
        assert len(input_img_shape) == 4
        bs, h, w, c = input_img_shape

        rep_img_shape = images.get_shape()
        assert len(rep_img_shape) == 2
        bs2, D = rep_img_shape
        bs.assert_is_compatible_with(bs2)

        jac_images = jac(images, self.images)
        jac_images.get_shape().assert_is_compatible_with([bs, h, w, c, D])
        jac_images = tf.reshape(jac_images, [bs, h * w * c, D])

        with tf.variable_scope('loss'):
            self.g_loss = kernel_sobolev_est(
                P_feats=G, Q_feats=images,
                mu_feats=images, jac_mu=jac_images,
                lam=self.hs, bw=1,
                mmd_unbiased=False,
                use_incho=self.config.use_incomplete_cho,
                incho_eta=self.config.incho_eta,
                incho_max=self.config.incho_max_steps)
            self.d_loss = -self.g_loss
            self.optim_name = 'kernelsobolev_loss'

        self.add_l2_penalty()


def jac(out, wrt):
    shp = out.get_shape()
    assert len(shp) == 2
    shp[0].assert_is_compatible_with(wrt.get_shape()[0])

    D = shp[1].value
    assert D is not None, "need explicit shape for output"

    return tf.concat([tf.expand_dims(tf.gradients(out[:, i], [wrt])[0], -1)
                      for i in range(D)], -1)


def kernel_sobolev_est(P_feats, Q_feats, mu_feats, jac_mu,
                       lam=1, bw=1, mmd_unbiased=False,
                       dtype=tf.float32,
                       use_incho=False, incho_eta=1e-3, incho_max=None):
    # NOTE: this currently only does a single Gaussian top-level kernel,
    #       though could support other isotropic shift-invariant kernels
    P_feats = tf.convert_to_tensor(P_feats, dtype=dtype, name="P_feats")
    Q_feats = tf.convert_to_tensor(Q_feats, dtype=dtype, name="Q_feats")
    mu_feats = tf.convert_to_tensor(mu_feats, dtype=dtype, name="mu_feats")
    lam = tf.convert_to_tensor(lam, dtype=dtype, name="lambda")
    bw = tf.convert_to_tensor(bw, dtype=dtype, name="bw")
    incho_eta = tf.convert_to_tensor(
        incho_eta, dtype=dtype, name="incho_eta")
    if incho_max is not None:
        incho_max = tf.convert_to_tensor(incho_max, dtype=tf.int32,
                                         name="incho_max")

    assert len(P_feats.get_shape()) == 2
    assert len(Q_feats.get_shape()) == 2
    assert len(mu_feats.get_shape()) == 2
    assert len(jac_mu.get_shape()) == 3

    n_P = tf.shape(P_feats)[0]
    n_Q = tf.shape(Q_feats)[0]
    n_delta = n_P + n_Q
    n_mu = tf.shape(mu_feats)[0]
    d = tf.shape(jac_mu)[1]
    gamma = 1 / (2 * bw**2)

    fn_P = tf.cast(n_P, dtype)
    fn_Q = tf.cast(n_Q, dtype)
    fn_mu = tf.cast(n_mu, dtype)

    # compute kernel matrix on delta samps
    PP = tf.matmul(P_feats, P_feats, transpose_b=True)
    PQ = tf.matmul(P_feats, Q_feats, transpose_b=True)
    QQ = tf.matmul(Q_feats, Q_feats, transpose_b=True)

    P_sqnorms = tf.diag_part(PP)
    Q_sqnorms = tf.diag_part(QQ)

    K_P = tf.exp(gamma * tf.minimum(
        0., 2 * PP - E(P_sqnorms, 0) - E(P_sqnorms, 1)))
    K_PQ = tf.exp(gamma * tf.minimum(
        0., 2 * PQ - E(P_sqnorms, 0) - E(Q_sqnorms, 1)))
    K_Q = tf.exp(gamma * tf.minimum(
        0., 2 * QQ - E(Q_sqnorms, 0) - E(Q_sqnorms, 1)))

    # first component: MMD estimator
    if mmd_unbiased:
        mmdsq_est = (
            (tf.reduce_sum(K_P) - fn_P) / (fn_P * (fn_P - 1))
            + (tf.reduce_sum(K_Q) - fn_Q) / (fn_Q * (fn_Q - 1))
        )
    else:
        mmdsq_est = tf.reduce_mean(K_P) + tf.reduce_mean(K_Q)
    mmdsq_est = mmdsq_est - 2 * tf.reduce_mean(K_PQ)

    # delta vector: diff in mean embeddings at X, and its gradients at X
    diff_P_mu = E(P_feats, 1) - E(mu_feats, 0)
    diff_Q_mu = E(Q_feats, 1) - E(mu_feats, 0)
    K_P_mu = tf.exp(-gamma * tf.reduce_sum(diff_P_mu ** 2, axis=2))
    K_Q_mu = tf.exp(-gamma * tf.reduce_sum(diff_Q_mu ** 2, axis=2))
    grad_K_P_mu = tf.reshape(
        2 * gamma * E(K_P_mu, 2) * tf.einsum('xyd,yjd->xyj', diff_P_mu, jac_mu),
        (n_P, n_mu * d))
    grad_K_Q_mu = tf.reshape(
        2 * gamma * E(K_Q_mu, 2) * tf.einsum('xyd,yjd->xyj', diff_P_mu, jac_mu),
        (n_P, n_mu * d))

    delta_at_mu = (tf.reduce_mean(K_P_mu, axis=0)
                   - tf.reduce_mean(K_Q_mu, axis=0))
    grad_delta_at_mu = (tf.reduce_mean(grad_K_P_mu, axis=0)
                        - tf.reduce_mean(grad_K_Q_mu, axis=0))
    delta_vec = tf.concat([delta_at_mu, grad_delta_at_mu], 0)

    # these are needed in constructing mu derivatives:
    diff_mu = tf.expand_dims(mu_feats, 1) - tf.expand_dims(mu_feats, 0)
    # diff_jac[x, y, i]  = (\phi(x) - \phi(y))^T \partial_{x_i} phi(x)
    #                    = diff[x, y, :] @ jac[x, i, :]
    # -diff_jac[y, x, i] = (\phi(x) - \phi(y))^T \partial_{y_j} phi(y)
    diff_jac_mu = tf.einsum('xyd,xid->xyi', diff_mu, jac_mu)

    if not use_incho:
        # construct the full [K G^T; G H] matrix
        K_mu = tf.exp(-gamma * tf.reduce_sum(diff_mu ** 2, axis=2))

        G_mu = tf.reshape(
            -2 * gamma * tf.expand_dims(K_mu, 1)
               * tf.einsum('xyd,xid->xiy', diff_mu, jac_mu),
            (n_mu * d, n_mu))

        H_mu = tf.reshape(
            2 * gamma * tf.expand_dims(tf.expand_dims(K_mu, 1), 3) * (
                tf.einsum('xid,yjd->xiyj', jac_mu, jac_mu)
                # sign flip because second arg should be -diff_jac_mu
                + 2 * gamma * tf.einsum('xyi,yxj->xiyj',
                                        diff_jac_mu, diff_jac_mu)),
            (n_mu * d, n_mu * d))

        together = tf.concat([
            tf.concat([K_mu, tf.transpose(G_mu)], 1),
            tf.concat([G_mu, H_mu], 1),
        ], 0)

        to_inv = together + fn_mu * lam * tf.eye(n_mu * (1 + d))
        pen_est = tf.squeeze(
            E(delta_vec, 0) @ tf.matrix_solve(to_inv, E(delta_vec, 1)),
            axis=[0, 1])

    else:
        # diagonal of H: diff_mu = 0 => diff_jac_mu = 0
        diag = tf.concat([
            tf.tile([1.], [n_mu]),
            2 * gamma * tf.reshape(tf.einsum('xid,xid->xi', jac_mu, jac_mu),
                                   [n_mu * d]),
        ], 0)

        def kgh_row(ind):
            # compute indth row of the [K G^T; G H] matrix
            # no python logic!

            # a row of [K G^T]
            def first_block():
                diff_a = diff_mu[ind]
                K_a = tf.exp(-gamma * tf.reduce_sum(diff_a ** 2, axis=-1))

                # ath column of G:
                #  (-2gamma E(K_mu,       1) einsum('xyd,xid->xiy', diff_mu, jac_mu))[:, :, a]
                # = -2gamma E(K_mu[:, a], 1) einsum('xd,xid->xi',   diff_mu[:, a, :], jac_mu)
                # = -2gamma E(K_mu[a, :], 1) einsum('xd,xid->xi',  -diff_mu[a, :, :], jac_mu)
                # =  2gamma E(K_mu[a],    1) einsum('xd,xid->xi',   diff_mu[a],       jac_mu)
                GT_a = tf.reshape(
                    2 * gamma * tf.expand_dims(K_a, 1)
                      * tf.einsum('xd,xid->xi', diff_a, jac_mu),
                    [n_mu * d])

                return tf.concat([K_a, GT_a], 0)

            # a row of [G H]
            def second_block():
                ind_pair = ind - n_mu
                a = ind_pair // d
                i = ind_pair % d

                diff_a = diff_mu[a]
                K_a = tf.exp(-gamma * tf.reduce_sum(diff_a ** 2, axis=-1))
                jac_ai = jac_mu[a, i, :]
                diff_jac_a = diff_jac_mu[a]

                # (a,i) row of G:
                #  (-2gamma E(K_mu, 1) einsum('xyd,xid->xiy', diff_mu, jac_mu))[a, i, :]
                # = -2gamma  K_mu[a]   einsum('yd,d->y', diff_mu[a], jac_mu[a, i, :])
                G_ind = -2 * gamma * K_a * tf.einsum('yd,d->y', diff_a, jac_ai)

                # (a,i) row of H:
                #   2 gamma E(E(K_mu, 1), 3) * (
                #       einsum('xid,yjd->xiyj', jac_mu, jac_mu)
                #     + 2gamma einsum('xyi,yxj->xiyj',
                #                     diff_jac_mu, diff_jac_mu))[a, i, :]
                # = 2 gamma E(K_mu[a, :], 1) * (
                #       einsum('xid,yjd->xiyj', jac_mu, jac_mu)[a, i, :]
                #     + 2 * gamma * einsum('xyi,yxj->xiyj', diff_jac_mu,
                #                          diff_jac_mu)[a, i, :])
                # = 2 gamma E(K_mu[a, :], 1) * (
                #       einsum('d,yjd->yj', jac_mu[a, i, :], jac_mu)
                #     + 2 * gamma * einsum('y,yj->yj', diff_jac_mu[a, :, i],
                #                          diff_jac_mu[:, a, :]))
                # = 2 gamma E(K_mu[a, :], 1) * (
                #       einsum('d,yjd->yj', jac_mu[a, i, :], jac_mu)
                #     - 2 * gamma * einsum('y,yj->yj', diff_jac_mu[a, :, i],
                #                          diff_jac_mu[a, :, :]))
                H_ind = tf.reshape(
                    2 * gamma * tf.expand_dims(K_a, 1) * (
                        tf.einsum('d,yjd->yj', jac_ai, jac_mu)
                        - 2 * gamma * tf.einsum('y,yj->yj',
                                                diff_jac_a[:, i], diff_jac_a)),
                    [n_mu * d])

                return tf.concat([G_ind, H_ind], 0)

            return tf.cond(ind < n_mu, first_block, second_block)

        R = incomplete_cholesky(diag, kgh_row, eta=incho_eta, max_T=incho_max)
        T = tf.shape(R)[0]
        tf.summary.scalar('cholesky_iters', T)
        R_delta = R @ E(delta_vec, 1)
        inner = tf.matmul(R, R, transpose_b=True) + fn_mu * lam * tf.eye(T)
        solved = tf.matrix_solve(inner, R_delta)
        pen_est = (
            2 * tf.nn.l2_loss(delta_vec)
            - tf.squeeze(tf.matmul(R_delta, solved, transpose_a=True),
                         axis=[0, 1])) / (fn_mu * lam)

    tf.summary.scalar("mmd_sq", mmdsq_est)
    tf.summary.scalar("penalty", pen_est)
    return (mmdsq_est - pen_est) / lam


def incomplete_cholesky(diag, compute_row, eta=0, max_T=None, scale_eta=True,
                        ret_inds=False):
    n = tf.shape(diag)[0]

    # first iteration:
    i0 = tf.argmax(diag, output_type=tf.int32)
    a = diag[i0]
    R = tf.expand_dims(compute_row(i0), 0) / tf.sqrt(a)
    diag = diag - R[0] ** 2

    if scale_eta:
        eta = eta * a

    # set up for second iteration:
    i = tf.argmax(diag, output_type=tf.int32)
    a = diag[i]
    if ret_inds:
        inds = tf.concat([tf.expand_dims(i0, 0), tf.expand_dims(i, 0)], 0)
        nu = tf.expand_dims(tf.sqrt(a), 0)
    else:
        inds = nu = tf.constant([0])

    def cond(diag, R, inds, nu, a, i):
        return a > eta

    def step(diag, R, inds, nu, a, i):
        sqrt_a = tf.sqrt(a)

        # for i in range(ell):
        #    R[j, i] = (K[I[j], i] - R[:, i].T @ R[:, I[j]]) / nu[j]
        b = tf.squeeze(tf.matmul(R, tf.expand_dims(R[:, i], 1),
                                 transpose_a=True), axis=1)
        R_row = (compute_row(i) - b) / sqrt_a

        R = tf.concat([R, tf.expand_dims(R_row, 0)], axis=0)
        diag = diag - R_row ** 2

        i = tf.argmax(diag, axis=0, output_type=tf.int32)
        if ret_inds:
            inds = tf.concat([inds, tf.expand_dims(i, 0)], 0)
            nu = tf.concat([nu, tf.expand_dims(sqrt_a, 0)], 0)
        a = diag[i]
        return diag, R, inds, nu, a, i

    maxit = n if max_T is None else tf.minimum(max_T, n)

    vec_shape = tf.TensorShape([None])
    R_shape = tf.TensorShape([None, diag.get_shape()[0]])

    diag, R, inds, nu, a, i = tf.while_loop(
        cond, step,
        loop_vars=(diag, R, inds, nu, a, i),
        shape_invariants=(diag.get_shape(), R_shape, vec_shape,
                          vec_shape, a.get_shape(), i0.get_shape()),
        maximum_iterations=maxit - 1)

    return (R, inds, nu) if ret_inds else R
