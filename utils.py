r"""Define probability distribution functions to compute 
lower bound on model evidence.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import shutil
import numpy as np

import torch
import torch.nn.functional as F

LOG2PI = np.log(2.0 * math.pi)


def save_checkpoint(state, is_best, folder='./',
                    filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


class AverageMeter(object):
    r"""Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unit_gaussian_to_gaussian(mu, logvar):
    # gaussian reparameterization trick
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def sample_from_unit_gaussian(device, batch_size, n_samples, z_dim):
    z = torch.Tensor(batch_size, n_samples, z_dim).normal_()
    z = z.to(device)
    return z


def unit_gaussian_params(batch_size, z_dim, device):
    mu = torch.zeros((batch_size, z_dim))
    logvar = torch.zeros((batch_size,  z_dim))
    mu, logvar = mu.to(device), logvar.to(device)

    return mu, logvar


def bernoulli_log_pdf(x, mu):
    r"""Log-likelihood of data given ~Bernoulli(mu)

    @param x: PyTorch.Tensor
              ground truth input
    @param mu: PyTorch.Tensor
               Bernoulli distribution parameters
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


def logistic_256_log_pdf(x, mean, logvar):
    r"""In practice it is problematic to use a gaussian decoder b/c it will
    memorize the data (defaulting to a regular decoder). Constraining it as
    a discrete space is important.

    https://www.reddit.com/r/MachineLearning/comments/4eqifs/gaussian_observation_vae/
    """
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = torch.log(cdf_plus - cdf_minus + 1.e-7)
    log_pdf = torch.sum(log_logist_256, 1)

    return log_pdf


def gaussian_log_pdf(x, mu, logvar):
    r"""Log-likelihood of data given ~N(mu, exp(logvar))

    log f(x) = log(1/sqrt(2*pi*var) * e^(-(x - mu)^2 / var))
             = -1/2 log(2*pi*var) - 1/2 * ((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2log(var) - 1/2((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2[((x-mu)/sigma)^2 + log var]

    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -LOG2PI * x.size(1) / 2. - \
        torch.sum(logvar + torch.pow(x - mu, 2) / (torch.exp(logvar) + 1e-7), dim=1) / 2.

    return log_pdf


def unit_gaussian_log_pdf(x):
    r"""Log-likelihood of data given ~N(0, 1)

    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -LOG2PI * x.size(1) / 2. - \
        torch.sum(torch.pow(x, 2), dim=1) / 2.

    return log_pdf


def log_mean_exp(x, dim=1):
    r"""log(1/k * sum(exp(x))): this normalizes x.

    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def flatten_and_permute_latents(z):
    batch_size, n_samples, z_dim = z.size()
    z2d = z.permute(0, 2, 1).contiguous()
    z2d = z2d.view(batch_size * z_dim, n_samples)
    
    return z2d.contiguous()


def unflatten_and_unpermute_latents(z2d, batch_size):
    z_dim = int(z2d.size(0) / batch_size)
    n_samples = z2d.size(1)
    z = z2d.view(batch_size, z_dim, n_samples)
    z = z.permute(0, 2, 1)
    
    return z.contiguous()
