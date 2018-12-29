from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .elbo import (
    pick_objective,
    log_bernoulli_marginal_estimate,
    log_bernoulli_norm_flow_marginal_estimate,
    log_bernoulli_volume_flow_marginal_estimate,
    log_logistic_marginal_estimate,
    log_logistic_norm_flow_marginal_estimate,
    log_logistic_volume_flow_marginal_estimate,
)
from .utils import unit_gaussian_to_gaussian
from .utils import sample_from_unit_gaussian, unit_gaussian_params
from .flows import NormalizingFlows, VolumePreservingFlows


def build_model(name, input_dim, z_dim, n_samples, hidden_dim=300,
                n_norm_flows=0, n_volume_flows=0, data_dist='bernoulli',
                **kwargs):
    r"""Helper function to pick the right model.

    @param name: string
                 vanilla|cheng|marsaglia
    """
    if name == 'vanilla':
        model = VAE(input_dim, z_dim, n_samples, hidden_dim=hidden_dim,
                    n_norm_flows=n_norm_flows, n_volume_flows=n_volume_flows, 
                    data_dist=data_dist)
    elif name == 'cheng':
        backprop = kwargs.get('backprop', False)
        model = ChengVAE(input_dim, z_dim, n_samples, hidden_dim=hidden_dim,
                         n_norm_flows=n_norm_flows, n_volume_flows=n_volume_flows, 
                         data_dist=data_dist, backprop=backprop)
    elif name == 'marsaglia':
        backprop = kwargs.get('backprop', False)
        model = MarsagliaVAE(input_dim, z_dim, n_samples, hidden_dim=hidden_dim,
                             n_norm_flows=n_norm_flows, n_volume_flows=n_volume_flows, 
                             data_dist=data_dist, backprop=backprop)
    else:
        raise Exception('model %s not supported.' % name)
    
    return model


class VAE(nn.Module):
    r"""Variational Autoencoder.

    Both the encoder and decoder are parameterized by MLPs.

    @param input_dim: integer
                      input example dimensionality
                      Example: 784 for MNIST
    @param z_dim: integer
                  number of latent dimensions.
    @param n_samples: integer
                      number of samples to take
    @param hidden_dim: integer [default: 300]
                       number of hidden dimensions
    @param n_norm_flows: integer [default: 0]
                         number of normalizing planar flows to apply
    @param n_volume_flows: integer [default: 0]
                           number of volume preserving flows to apply
    @param data_dist: string [default: bernoulli]
                      distribution of data
                      bernoulli|logistic
    """
    def __init__(self, input_dim, z_dim, n_samples, hidden_dim=300,
                 n_norm_flows=0, n_volume_flows=0, data_dist='bernoulli'):
        super(VAE, self).__init__()
        assert not (n_norm_flows > 0 and n_volume_flows > 0), \
            "cannot have both normalizing and volume preserving flows."
        assert data_dist in ['bernoulli', 'logistic'], \
            "data_dist must be bernoulli|logistic."
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_norm_flows = n_norm_flows
        self.n_volume_flows = n_volume_flows
        self.data_dist = data_dist

        self.encoder = GaussianEncoder( self.input_dim, self.z_dim, 
                                        hidden_dim=self.hidden_dim)

        if self.data_dist == 'bernoulli':
            self.decoder = BernoulliDecoder(
                self.input_dim, self.z_dim, hidden_dim=hidden_dim)
        else:
            self.decoder = GaussianDecoder(
                self.input_dim, self.z_dim, hidden_dim=hidden_dim)

        if self.n_norm_flows > 0:
            self.norm_flows = NormalizingFlows(self.z_dim, n_flows=self.n_norm_flows)

        if self.n_volume_flows > 0:
            self.volume_flows = VolumePreservingFlows(self.z_dim, n_flows=self.n_volume_flows)

        # i heard xavier init is good
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _reparameterize(self, mu, logvar):
        z = unit_gaussian_to_gaussian(mu, logvar)
        return z
    
    def reparameterize(self, mu, logvar):
        return self._reparameterize(mu, logvar)

    def forward(self, x):
        batch_size = x.size(0)
        z0_mu, z0_logvar, z0_h = self.encoder(x)  # z0_h is only for VP-flows

        # add a dimension for multiple samples
        z0_mu = z0_mu.unsqueeze(1).repeat(1, self.n_samples, 1)
        z0_logvar = z0_logvar.unsqueeze(1).repeat(1, self.n_samples, 1)

        # important for this to be reparmaeterize, not _reparameterize
        z0 = self.reparameterize(z0_mu, z0_logvar)
        z0_2d = z0.view(batch_size * self.n_samples, self.z_dim)

        if self.n_norm_flows > 0:
            zk_2d, log_abs_det_jacobian2d = self.norm_flows(z0_2d)
            zk = zk_2d.view(batch_size, self.n_samples, self.z_dim)
            log_abs_det_jacobian = log_abs_det_jacobian2d.view(batch_size, self.n_samples)

            if self.data_dist == 'bernoulli':
                recon_x_mu2d = self.decoder(zk_2d)
                recon_x_mu = recon_x_mu2d.view(batch_size, self.n_samples, self.input_dim)
                return recon_x_mu, zk, z0, z0_mu, z0_logvar, log_abs_det_jacobian
            else:  # logistic
                recon_x_mu2d, recon_x_logvar2d = self.decoder(zk_2d)
                recon_x_mu = recon_x_mu2d.view(batch_size, self.n_samples, self.input_dim)
                recon_x_logvar = recon_x_logvar2d.view(batch_size, self.n_samples, self.input_dim)
                return recon_x_mu, recon_x_logvar, zk, z0, z0_mu, z0_logvar, log_abs_det_jacobian

        elif self.n_volume_flows > 0:
            z0_h = z0_h.unsqueeze(1).repeat(1, self.n_samples, 1)
            z0_h_2d = z0_h.view(batch_size * self.n_samples, self.hidden_dim)

            zk_2d = self.volume_flows(z0_2d, z0_h_2d)
            zk = zk_2d.view(batch_size, self.n_samples, self.z_dim)

            if self.data_dist == 'bernoulli':
                recon_x_mu2d = self.decoder(zk_2d)
                recon_x_mu = recon_x_mu2d.view(batch_size, self.n_samples, self.input_dim)
                return recon_x_mu, zk, z0, z0_mu, z0_logvar
            else:  # logistic
                recon_x_mu2d, recon_x_logvar2d = self.decoder(zk_2d)
                recon_x_mu = recon_x_mu2d.view(batch_size, self.n_samples, self.input_dim)
                recon_x_logvar = recon_x_logvar2d.view(batch_size, self.n_samples, self.input_dim)
                return recon_x_mu, recon_x_logvar, zk, z0, z0_mu, z0_logvar

        else:
            if self.data_dist == 'bernoulli':
                recon_x_mu2d = self.decoder(z0_2d)
                recon_x_mu = recon_x_mu2d.view(batch_size, self.n_samples, self.input_dim)
                return recon_x_mu, z0, z0_mu, z0_logvar
            else:  # logistic
                recon_x_mu2d, recon_x_logvar2d = self.decoder(z0_2d)
                recon_x_mu = recon_x_mu2d.view(batch_size, self.n_samples, self.input_dim)
                recon_x_logvar = recon_x_logvar2d.view(batch_size, self.n_samples, self.input_dim)
                return recon_x_mu, recon_x_logvar, z0, z0_mu, z0_logvar

    def estimate_marginal(self, x, n_samples=100):
        # Monte Carlo estimate of log p(x)
        # this is not to be used for an objective
        batch_size = x.size(0)
        z0_mu, z0_logvar, z0_h = self.encoder(x)

        zk, z0, recon_x_mu, recon_x_logvar = [], [], [], []
        log_abs_det_jacobian = []

        for i in range(n_samples):
            # important for this to be _reparameterize, not reparameterize
            z0_i = self._reparameterize(z0_mu, z0_logvar)
            z0.append(z0_i)

            if self.n_norm_flows > 0:
                zk_i, log_abs_det_jacobian_i = self.norm_flows(z0_i)
                if self.data_dist == 'bernoulli':
                    recon_x_mu_i = self.decoder(zk_i)
                    recon_x_mu.append(recon_x_mu_i)
                else:  # logistic
                    recon_x_mu_ i, recon_x_logvar_i = self.decoder(zk_i)
                    recon_x_mu.append(recon_x_mu_i)
                    recon_x_logvar.append(recon_x_logvar_i)
                zk.append(zk_i)
                log_abs_det_jacobian.append(log_abs_det_jacobian_i)
            elif self.n_volume_flows > 0:
                zk_i = self.volume_flows(z0_i, z0_h)
                if self.data_dist == 'bernoulli':
                    recon_x_mu_i = self.decoder(zk_i)
                    recon_x_mu.append(recon_x_mu_i)
                else:  # logistic
                    recon_x_mu_i, recon_x_logvar_i = self.decoder(zk_i)
                    recon_x_mu.append(recon_x_mu_i)
                    recon_x_logvar.append(recon_x_logvar_i)
                zk.append(zk_i)
            else:
                if self.data_dist == 'bernoulli':
                    recon_x_mu_i = self.decoder(z0_i)
                    recon_x_mu.append(recon_x_mu_i)
                else:  # logistic
                    recon_x_mu_i, recon_x_logvar_i = self.decoder(z0_i)
                    recon_x_mu.append(recon_x_mu_i)
                    recon_x_logvar.append(recon_x_logvar_i)

        z0 = torch.stack(z0).permute(1, 0, 2).contiguous()
        z0_mu = z0_mu.unsqueeze(1).repeat(1, n_samples, 1)
        z0_logvar = z0_logvar.unsqueeze(1).repeat(1, n_samples, 1)
        recon_x_mu = torch.stack(recon_x_mu).permute(1, 0, 2).contiguous()

        if self.n_norm_flows > 0:
            zk = torch.stack(zk).permute(1, 0, 2).contiguous()
            log_abs_det_jacobian = torch.stack(log_abs_det_jacobian).t().contiguous()
        elif self.n_volume_flows > 0:
            zk = torch.stack(zk).permute(1, 0, 2).contiguous()

        if self.data_dist != 'bernoulli':
            recon_x_logvar = torch.stack(recon_x_logvar).permute(1, 0, 2).contiguous()

        if self.n_norm_flows > 0:
            if self.data_dist == 'bernoulli':
                return log_bernoulli_norm_flow_marginal_estimate(
                    recon_x_mu, x, zk, z0, z0_mu, z0_logvar, log_abs_det_jacobian)
            else:  # logistic
                return log_logistic_norm_flow_marginal_estimate(
                    recon_x_mu, recon_x_logvar, x, zk, z0, z0_mu, z0_logvar,
                    log_abs_det_jacobian)
        elif self.n_volume_flows > 0:
            if self.data_dist == 'bernoulli':
                return log_bernoulli_volume_flow_marginal_estimate(
                    recon_x_mu, x, zk, z0, z0_mu, z0_logvar)
            else:  # logistic
                return log_logistic_volume_flow_marginal_estimate(
                    recon_x_mu, recon_x_logvar, x, zk, z0, z0_mu, z0_logvar)
        else:
            if self.data_dist == 'bernoulli':
                return log_bernoulli_marginal_estimate(recon_x_mu, x, z0, z0_mu, z0_logvar)
            else:  # logistic
                return log_logistic_marginal_estimate(
                    recon_x_mu, recon_x_logvar, x, z0, z0_mu, z0_logvar)


class ChengVAE(VAE):
    r"""Antithetic Variational Autoencoder.

    Use 1-dimensional Cheng's algorithm to draw antithetic
    samples. Also, use Hilfery's approximation instead of
    computing the inverse distribution function transformation.

    @param input_dim: integer
                      input example dimensionality
                      Example: 784 for MNIST
    @param z_dim: integer
                  number of latent dimensions.
    @param n_samples: integer
                      number of samples to take
                      (this must be a even number).
    @param hidden_dim: integer [default: 300]
                       number of hidden dimensions
    @param n_norm_flows: integer [default: 0]
                         number of normalizing planar flows to apply
    @param n_volume_flows: integer [default: 0]
                           number of volume preserving flows to apply
    @param backprop: boolean [default: True]
                     if True, backpropagate through Cheng
                     transform, otherwise, transform in
                     epsilon space only.
    """
    def __init__(self, input_dim, z_dim, n_samples, hidden_dim=300,
                 n_norm_flows=0, n_volume_flows=0, data_dist='bernoulli', 
                 backprop=True):
        super(ChengVAE, self).__init__(
            input_dim, z_dim, n_samples, hidden_dim=hidden_dim,
            n_norm_flows=n_norm_flows, n_volume_flows=n_volume_flows,
            data_dist=data_dist)
        assert n_samples >= 4, "Limitation of algorithm: n_samples >= 4."
        assert n_samples % 2 == 0, "n_samples must be even."
        self.backprop = backprop

    def reparameterize(self, mu, logvar):
        batch_size, device = mu.size(0), mu.device
        # randomly sample half of samples; other half will be derived
        k = self.n_samples // 2

        z = torch.normal(torch.zeros(batch_size, k - 1, self.z_dim),
                         torch.ones(batch_size, k - 1, self.z_dim))
        b = torch.bernoulli(torch.ones(batch_size, k - 1, self.z_dim) * 0.5))
        z, b = z.to(device), b.to(device)

        if self.backprop:
            x = self._reparameterize(mu[:, :k], logvar[:, :k])
            mu, logvar = mu[:, 0].contiguous(), logvar[:, 0].contiguous()
            sigma = torch.exp(2. * logvar)
            x_anti, _ = cheng_antithetic_nd(z, b, x, mu, sigma, k)
            x = torch.cat((x, x_anti), dim=1)
        else:
            r = sample_from_unit_gaussian(batch_size, k, self.z_dim, device)
            r_mu, r_logvar = unit_gaussian_params(batch_size, self.z_dim, device)
            r_sigma = torch.exp(2. * r_logvar)
            r_anti, _ = cheng_antithetic_nd(z, b, r, r_mu, r_sigma, k)
            r = torch.cat((r, r_anti), dim=1)
            x = unit_gaussian_to_gaussian(mu, logvar, eps=r)

        return x


class MarsagliaVAE(VAE):
    def __init__(self, input_dim, z_dim, n_samples, hidden_dim=300,
                 n_norm_flows=0, n_volume_flows=0, data_dist='bernoulli', backprop=True):
        super(MarsagliaVAE, self).__init__(
            input_dim, z_dim, n_samples, hidden_dim=hidden_dim,
            n_norm_flows=n_norm_flows, n_volume_flows=n_volume_flows,
            data_dist=data_dist)
        assert n_samples >= 4, "Limitation of algorithm: n_samples >= 4."
        assert n_samples % 2 == 0, "n_samples must be even."
        self.backprop = backprop

    def reparameterize(self, mu, logvar):
        batch_size, device = mu.size(0), mu.device
        # randomly sample half of samples; other half will be derived
        k = self.n_samples // 2

        z = torch.normal(torch.zeros(batch_size, k - 1, self.z_dim),
                         torch.ones(batch_size, k - 1, self.z_dim))
        z = z.to(device)

        if self.backprop:
            x = self._reparameterize(mu[:, :k], logvar[:, :k])
            mu, logvar = mu[:, 0].contiguous(), logvar[:, 0].contiguous()
            sigma = torch.exp(2. * logvar)
            x_anti = marsaglia_antithetic_nd(z, x, mu, sigma, k)
            x = torch.cat((x, x_anti), dim=1)
        else:
            r = sample_from_unit_gaussian(batch_size, k, self.z_dim, device)
            r_mu, r_logvar = unit_gaussian_params(batch_size, self.z_dim, device)
            r_sigma = torch.exp(2. * r_logvar)
            r_anti = marsaglia_antithetic_nd(z, r, r_mu, r_sigma, k)
            r = torch.cat((r, r_anti), dim=1)
            x = unit_gaussian_to_gaussian(mu, logvar, eps=r)

        return x


class GaussianEncoder(nn.Module):
    r"""Parametrizes q(z|x).

    Architecture design modeled after:
    https://github.com/jmtomczak/vae_vampprior

    @param input_dim: integer
                      number of input dimension.
    @param z_dim: integer
                  number of latent dimensions.
    @param hidden_dim: integer [default: 300]
                       number of hidden dimensions.
    """
    def __init__(self, input_dim, z_dim, hidden_dim=300):
        super(GaussianEncoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.z_dim * 2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h1))
        h2 = self.fc3(h1)
        mu, logvar = torch.chunk(h2, 2, dim=1)

        return mu, logvar, h1


class BernoulliDecoder(nn.Module):
    r"""Parametrizes p(x|z).

    Architecture design modeled after:
    https://github.com/jmtomczak/vae_vampprior

    @param input_dim: integer
                      number of input dimension.
    @param z_dim: integer
                  number of latent dimensions.
    @param hidden_dim: integer [default: 300]
                       number of hidden dimensions.
    """
    def __init__(self, input_dim, z_dim, hidden_dim=300):
        super(BernoulliDecoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        mu = torch.sigmoid(h)

        return mu  # parameters of bernoulli


class GaussianDecoder(nn.Module):
    r"""Parameterizes p(x|z) but using a gaussian distribution
    over the decoded variables.

    @param input_dim: integer
                      number of input dimensions
    @param z_dim: integer
                  number of latent dimensions.
    @param hidden_dim: integer [default: 300]
                       number of hidden dimensions.
    """
    def __init__(self, input_dim, z_dim, hidden_dim=300):
        super(GaussianDecoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.input_dim * 2)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        mu, logvar = mu.contiguous(), logvar.contiguous()

        # https://github.com/jmtomczak/vae_householder_flow/blob/9a71a42c494ad513bf679641fb5a192709d73681/models/vae_HF.py#L104
        # https://github.com/jmtomczak/vae_vampprior/blob/master/models/VAE.py#L46
        mu = torch.sigmoid(mu)
        # https://github.com/jmtomczak/vae_vampprior/blob/master/models/VAE.py#L47
        logvar = F.hardtanh(logvar, min_val=-4.5,max_val=0.)

        return mu, logvar
