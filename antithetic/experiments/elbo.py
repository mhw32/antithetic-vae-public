r"""Evidence Lower Bound on Marginal Log-Likelihoods.

Define a bunch of different kinds of ELBOs.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from .utils import (
    bernoulli_log_pdf,
    gaussian_log_pdf,
    unit_gaussian_log_pdf,
    logistic_256_log_pdf,
    log_mean_exp,
)


def build_objective(iwae=False, use_flow=False, bernoulli=True):
    r"""Helper function to pick the right objective!

    @param iwae: boolean [default: False]
                 use importance weighted lower bound 
    @param use_flow: boolean [default: False]
                     apply either normalizing or volume-preserving flows
    @param bernoulli: bernoulli [default: True]
                      distribution over data; if not bernoulli, then logistic
    """
    if bernoulli:
        if use_flow:
            assert not iwae
            return bernoulli_free_energy_bound
        else:
            if iwae:
                return weighted_bernoulli_elbo_loss
            else:
                return bernoulli_elbo_loss
    else:  # logistic
        if use_flow:
            assert not iwae
            return gaussian_free_energy_bound
        else:
            if iwae:
                return weighted_gaussian_elbo_loss
            else:
                return gaussian_elbo_loss
                

def bernoulli_elbo_loss(recon_x_mu, x, z, z_mu, z_logvar):
    r"""Lower bound on model evidence (average over multiple samples).

    Closed form solution for KL[p(z|x), p(z)]

    Kingma, Diederik P., and Max Welling. "Auto-encoding
    variational bayes." arXiv preprint arXiv:1312.6114 (2013).
    """
    n_samples = recon_x_mu.size(1)

    ELBO = 0
    for i in xrange(n_samples):
        BCE = -bernoulli_log_pdf(x, recon_x_mu[:, i])
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * (1 + z_logvar[:, i] - z_mu[:, i].pow(2) - z_logvar[:, i].exp())
        KLD = torch.sum(KLD, dim=1)

        # lower bound on marginal likelihood
        ELBO_i = BCE + KLD
        ELBO += ELBO_i

    ELBO = ELBO / float(n_samples)
    ELBO = torch.mean(ELBO)

    return ELBO


def weighted_bernoulli_elbo_loss(recon_x_mu, x, z, z_mu, z_logvar):
    r"""Importance weighted evidence lower bound.

    @param recon_x_mu: torch.Tensor (batch size x # samples x |input_dim|)
                       reconstructed means on bernoulli
    @param x: torch.Tensor (batch size x |input_dim|)
                 original observed data
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param z_mu: torch.Tensor (batch_size x # samples x z dim)
                 means of variational distribution
    @param z_logvar: torch.Tensor (batch_size x # samples x z dim)
                     log-variance of variational distribution
    """
    batch_size = recon_x_mu.size(0)
    n_samples = recon_x_mu.size(1)

    log_ws = []
    for i in xrange(n_samples):
        log_p_x_given_z = bernoulli_log_pdf(x, recon_x_mu[:, i])
        log_q_z_given_x = gaussian_log_pdf(z[:, i], z_mu[:, i], z_logvar[:, i])
        log_p_z = unit_gaussian_log_pdf(z[:, i])

        log_ws_i = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_ws.append(log_ws_i.unsqueeze(1))

    log_ws = torch.cat(log_ws, dim=1)
    log_ws = log_mean_exp(log_ws, dim=1)
    BOUND = -torch.mean(log_ws)

    return BOUND


def bernoulli_free_energy_bound(recon_x_mu, x, zk, z0, z_mu, z_logvar,
                                log_abs_det_jacobian=None, beta=1.):
    r"""Lower bound on approximate posterior distribution transformed by
    many normalizing flows.

    This uses the closed form solution for ELBO. See <closed_form_elbo_loss>
    for more details.

    See https://github.com/Lyusungwon/generative_models_pytorch/blob/master/vae_nf/main.py

    For volume preserving transformatinos, keep log_abs_det_jacobian as None.
    """
    assert z0.size() == zk.size()
    n_samples = recon_x_mu.size(1)

    BOUND = 0
    for i in xrange(n_samples):
        log_p_x_given_z = bernoulli_log_pdf(x, recon_x_mu[:, i])
        log_q_z0_given_x = gaussian_log_pdf(z0[:, i], z_mu[:, i], z_logvar[:, i])
        log_p_zk = unit_gaussian_log_pdf(zk[:, i])

        if log_abs_det_jacobian is not None:
            log_q_zk_given_x = log_q_z0_given_x - log_abs_det_jacobian[:, i]
        else:
            log_q_zk_given_x = log_q_z0_given_x

        BOUND_i = log_p_x_given_z + beta * (log_p_zk - log_q_zk_given_x)
        BOUND += BOUND_i

    BOUND = BOUND / float(n_samples)
    BOUND = -BOUND
    BOUND = torch.mean(BOUND)

    return BOUND


# --- similar to functions above but for gaussian distribution over x ---


def gaussian_elbo_loss(recon_x_mu, recon_x_logvar, x, z, z_mu, z_logvar):
    n_samples = recon_x_mu.size(1)

    ELBO = 0
    for i in xrange(n_samples):
        BCE = -logistic_256_log_pdf(x, recon_x_mu[:, i], recon_x_logvar[:, i])
        KLD = -0.5 * (1 + z_logvar[:, i] - z_mu[:, i].pow(2) - z_logvar[:, i].exp())
        KLD = torch.sum(KLD, dim=1)

        ELBO_i = BCE + KLD
        ELBO += ELBO_i

    ELBO = ELBO / float(n_samples)
    ELBO = torch.mean(ELBO)

    return ELBO


def weighted_gaussian_elbo_loss(recon_x_mu, recon_x_logvar, x, z, z_mu, z_logvar):
    n_samples = recon_x_mu.size(1)

    log_ws = []
    for i in xrange(n_samples):
        log_p_x_given_z = logistic_256_log_pdf(x, recon_x_mu[:, i], recon_x_logvar[:, i])
        log_q_z_given_x = gaussian_log_pdf(z[:, i], z_mu[:, i], z_logvar[:, i])
        log_p_z = unit_gaussian_log_pdf(z[:, i])

        log_ws_i = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_ws.append(log_ws_i.unsqueeze(1))

    log_ws = torch.cat(log_ws, dim=1)
    log_ws = log_mean_exp(log_ws, dim=1)
    BOUND = -torch.mean(log_ws)

    return BOUND


def gaussian_free_energy_bound(recon_x_mu, recon_x_logvar, x, zk, z0, z_mu, z_logvar,
                               log_abs_det_jacobian=None, beta=1.):
    assert z0.size() == zk.size()
    n_samples = recon_x_mu.size(1)

    BOUND = 0
    for i in xrange(n_samples):
        log_p_x_given_z = logistic_256_log_pdf(x, recon_x_mu[:, i], recon_x_logvar[:, i])
        log_q_z0_given_x = gaussian_log_pdf(z0[:, i], z_mu[:, i], z_logvar[:, i])
        log_p_zk = unit_gaussian_log_pdf(zk[:, i])

        if log_abs_det_jacobian is not None:
            log_q_zk_given_x = log_q_z0_given_x - log_abs_det_jacobian[:, i]
        else:
            log_q_zk_given_x = log_q_z0_given_x

        BOUND_i = log_p_x_given_z + beta * (log_p_zk - log_q_zk_given_x)
        BOUND += BOUND_i

    BOUND = BOUND / float(n_samples)
    BOUND = -BOUND
    BOUND = torch.mean(BOUND)

    return BOUND


# --- marginal log likelihood estimators ---


def log_bernoulli_marginal_estimate(recon_x_mu, x, z, z_mu, z_logvar):
    r"""Estimate log p(x). NOTE: this is not the objective that
    should be directly optimized.

    @param recon_x_mu: torch.Tensor (batch size x # samples x input_dim)
                       reconstructed means on bernoulli
    @param x: torch.Tensor (batch size x input_dim)
              original observed data
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param z_mu: torch.Tensor (batch_size x # samples x z dim)
                 means of variational distribution
    @param z_logvar: torch.Tensor (batch_size x # samples x z dim)
                     log-variance of variational distribution
    """
    batch_size, n_samples, z_dim = z.size()
    input_dim = x.size(1)
    x = x.unsqueeze(1).repeat(1, n_samples, 1)

    z_2d = z.view(batch_size * n_samples, z_dim)
    z_mu_2d = z_mu.view(batch_size * n_samples, z_dim)
    z_logvar_2d = z_logvar.view(batch_size * n_samples, z_dim)
    recon_x_mu_2d = recon_x_mu.view(batch_size * n_samples, input_dim)
    x_2d = x.view(batch_size * n_samples, input_dim)

    log_p_x_given_z_2d = bernoulli_log_pdf(x_2d, recon_x_mu_2d)
    log_q_z_given_x_2d = gaussian_log_pdf(z_2d, z_mu_2d, z_logvar_2d)
    log_p_z_2d = unit_gaussian_log_pdf(z_2d)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p_x = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p_x)


def log_bernoulli_norm_flow_marginal_estimate(recon_x_mu, x, zk, z0, z0_mu, z0_logvar, log_abs_det_jacobian):
    batch_size, n_samples, z_dim = z0.size()
    input_dim = x.size(1)
    x = x.unsqueeze(1).repeat(1, n_samples, 1)

    z0_2d = z0.view(batch_size * n_samples, z_dim)
    zk_2d = zk.view(batch_size * n_samples, z_dim)
    z0_mu_2d = z0_mu.view(batch_size * n_samples, z_dim)
    z0_logvar_2d = z0_logvar.view(batch_size * n_samples, z_dim)
    log_abs_det_jacobian_2d = \
        log_abs_det_jacobian.view(batch_size * n_samples)
    recon_x_mu_2d = recon_x_mu.view(batch_size * n_samples, input_dim)
    x_2d = x.view(batch_size * n_samples, input_dim)

    log_p_x_given_zk_2d = bernoulli_log_pdf(x_2d, recon_x_mu_2d)
    log_q_z0_given_x_2d = gaussian_log_pdf(z0_2d, z0_mu_2d, z0_logvar_2d)
    log_q_zk_given_x_2d = log_q_z0_given_x_2d - log_abs_det_jacobian_2d
    log_p_zk_2d = unit_gaussian_log_pdf(zk_2d)

    log_weight_2d = log_p_x_given_zk_2d + log_p_zk_2d - log_q_zk_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    log_p_x = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p_x)


def log_bernoulli_volume_flow_marginal_estimate(recon_x_mu, x, zk, z0, z0_mu, z0_logvar):
    batch_size, n_samples, z_dim = z0.size()
    input_dim = x.size(1)
    x = x.unsqueeze(1).repeat(1, n_samples, 1)

    z0_2d = z0.view(batch_size * n_samples, z_dim)
    zk_2d = zk.view(batch_size * n_samples, z_dim)
    z0_mu_2d = z0_mu.view(batch_size * n_samples, z_dim)
    z0_logvar_2d = z0_logvar.view(batch_size * n_samples, z_dim)
    recon_x_mu_2d = recon_x_mu.view(batch_size * n_samples, input_dim)
    x_2d = x.view(batch_size * n_samples, input_dim)

    log_p_x_given_zk_2d = bernoulli_log_pdf(x_2d, recon_x_mu_2d)
    log_q_z0_given_x_2d = gaussian_log_pdf(z0_2d, z0_mu_2d, z0_logvar_2d)
    log_q_zk_given_x_2d = log_q_z0_given_x_2d  # diff
    log_p_zk_2d = unit_gaussian_log_pdf(zk_2d)

    log_weight_2d = log_p_x_given_zk_2d + log_p_zk_2d - log_q_zk_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    log_p_x = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p_x)


def log_logistic_marginal_estimate(recon_x_mu, recon_x_logvar, x, z, z_mu, z_logvar):
    batch_size, n_samples, z_dim = z.size()
    input_dim = x.size(1)
    x = x.unsqueeze(1).repeat(1, n_samples, 1)

    z_2d = z.view(batch_size * n_samples, z_dim)
    z_mu_2d = z_mu.view(batch_size * n_samples, z_dim)
    z_logvar_2d = z_logvar.view(batch_size * n_samples, z_dim)
    recon_x_mu_2d = recon_x_mu.view(batch_size * n_samples, input_dim)
    recon_x_logvar_2d = recon_x_logvar.view(batch_size * n_samples, input_dim)
    x_2d = x.view(batch_size * n_samples, input_dim)

    log_p_x_given_z_2d = logistic_256_log_pdf(x_2d, recon_x_mu_2d, recon_x_logvar_2d)
    log_q_z_given_x_2d = gaussian_log_pdf(z_2d, z_mu_2d, z_logvar_2d)
    log_p_z_2d = unit_gaussian_log_pdf(z_2d)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    log_p_x = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p_x)


def log_logistic_norm_flow_marginal_estimate(recon_x_mu, recon_x_logvar, x, zk, z0, z0_mu, z0_logvar, log_abs_det_jacobian):
    batch_size, n_samples, z_dim = z0.size()
    input_dim = x.size(1)
    x = x.unsqueeze(1).repeat(1, n_samples, 1)

    z0_2d = z0.view(batch_size * n_samples, z_dim)
    zk_2d = zk.view(batch_size * n_samples, z_dim)
    z0_mu_2d = z0_mu.view(batch_size * n_samples, z_dim)
    z0_logvar_2d = z0_logvar.view(batch_size * n_samples, z_dim)
    log_abs_det_jacobian_2d = \
        log_abs_det_jacobian.view(batch_size * n_samples)
    recon_x_mu_2d = recon_x_mu.view(batch_size * n_samples, input_dim)
    recon_x_logvar_2d = recon_x_logvar.view(batch_size * n_samples, input_dim)
    x_2d = x.view(batch_size * n_samples, input_dim)

    log_p_x_given_zk_2d = logistic_256_log_pdf(x_2d, recon_x_mu_2d, recon_x_logvar_2d)
    log_q_z0_given_x_2d = gaussian_log_pdf(z0_2d, z0_mu_2d, z0_logvar_2d)
    log_q_zk_given_x_2d = log_q_z0_given_x_2d - log_abs_det_jacobian_2d
    log_p_zk_2d = unit_gaussian_log_pdf(zk_2d)

    log_weight_2d = log_p_x_given_zk_2d + log_p_zk_2d - log_q_zk_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    log_p_x = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p_x)


def log_logistic_volume_flow_marginal_estimate(recon_x_mu, recon_x_logvar, x, zk, z0, z0_mu, z0_logvar):
    batch_size, n_samples, z_dim = z0.size()
    input_dim = x.size(1)
    x = x.unsqueeze(1).repeat(1, n_samples, 1)

    z0_2d = z0.view(batch_size * n_samples, z_dim)
    zk_2d = zk.view(batch_size * n_samples, z_dim)
    z0_mu_2d = z0_mu.view(batch_size * n_samples, z_dim)
    z0_logvar_2d = z0_logvar.view(batch_size * n_samples, z_dim)
    recon_x_mu_2d = recon_x_mu.view(batch_size * n_samples, input_dim)
    recon_x_logvar_2d = recon_x_logvar.view(batch_size * n_samples, input_dim)
    x_2d = x.view(batch_size * n_samples, input_dim)

    log_p_x_given_zk_2d = logistic_256_log_pdf(x_2d, recon_x_mu_2d, recon_x_logvar_2d)
    log_q_z0_given_x_2d = gaussian_log_pdf(z0_2d, z0_mu_2d, z0_logvar_2d)
    log_q_zk_given_x_2d = log_q_z0_given_x_2d  # diff
    log_p_zk_2d = unit_gaussian_log_pdf(zk_2d)

    log_weight_2d = log_p_x_given_zk_2d + log_p_zk_2d - log_q_zk_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    log_p_x = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p_x)
