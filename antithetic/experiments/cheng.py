r"""Cheng transformations to generate univariate samples
with a given sample mean and variance. Implemented in PyTorch.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np

import torch
from torch.autograd import Variable

from .utils import (
    flatten_and_permute_latents,
    unflatten_and_unpermute_latents,
)


def cheng_sample_1d(z, b, mu, sigma, eta, delta, k):
    r"""Given k-1 i.i.d. samples z1, ..., z_{k-1} \sim N(0, 1);
    k_1 i.i.d samples b1, ..., bk_1 \sim Bern(0.5)
    population moments from a Gaussian distribution mu, sigma^2
    desired sample moments eta, delta^2 such that eta \sim N(mu, sigma^2/k)
    and (k-1)delta^2/sigma^2 \sim chi^2(k-1), then the generated samples
    (x_1, ..., x_k) \sim N(mu, sigma^2) with sample mean eta and
    sample variance delta^2.

    Additionally intermediatary values:
        y_1 = sqrt(k)(eta - mu)
        sum(y_2_to_k^2) = sum((x - mean(x))^2)

    @param z: torch.Tensor
              shape: batch_size x k - 1
              samples from unit Gaussian, N(0, 1)
    @param b: torch.Tensor
              shape: batch_size x k - 1
              samples from Bernoulli
    @param mu: torch.Tensor
               shape: batch_size
               population means
    @param sigma: torch.Tensor
                  shape: batch_size
                  population standard deviations
    @param eta: torch.Tensor
                shape: batch_size
                sample means
    @param delta: torch.Tensor
                  shape: batch_size
                  sample standard deviations
    @return x: torch.Tensor
               shape: batch_size x k
               samples with desired properties
    @return y: torch.Tensor
               shape: batch_size x k
               samples prior to Helmert (for sanity checks)
    """
    assert z.size(1) == (k - 1)
    assert b.size(1) == (k - 1)

    c = torch.pow(z * unsqueeze_and_repeat(sigma, k - 1), 2)
    a = (k - 1) * torch.pow(delta, 2) / torch.sum(c, dim=1)
    y_squared_2_to_k = a.unsqueeze(1) * c
    y_2_to_k = (2 * b - 1) * torch.sqrt(y_squared_2_to_k + 1e-15)
    y_1 = math.sqrt(k) * (eta - mu)
    y = torch.cat([y_1.unsqueeze(1), y_2_to_k], dim=1)

    x_1 = 1./k * (k * eta - math.sqrt(k * (k - 1)) * y[:, 1])
    x = [x_1]
    for _j in range(1, k):
        j = _j + 1
        if j == k:
            x_j = x[_j - 1] + math.pow(k + 1 - j, -0.5) * (math.pow(k + 2 - j, 0.5) * y[:, _j])
        else:
            x_j = x[_j - 1] + math.pow(k + 1 - j, -0.5) * (math.pow(k + 2 - j, 0.5) * y[:, _j] - math.pow(k - j, 0.5) * y[:, _j + 1])
        x.append(x_j)

    x = torch.stack(x).t().contiguous()

    return x, y


def cheng_sample_nd(z, b, mu, sigma, eta, delta, k):
    r"""Treat n dimensions as independent samples. We flatten
    each of these inputs to be n*k. See <cheng_sample_1d> for
    more details.

    Shape of z and b tensors
        batch_size x (k - 1) x z_dim

    Shape of mu, sigma, eta, delta
        batch_size x z_dim

    The returned samples will be of the size:
        batch_size x k x z_dim
    """
    assert z.size(1) == (k - 1)
    assert b.size(1) == (k - 1)

    batch_size, _, z_dim = z.size()
    z2d = flatten_and_permute_latents(z)
    b2d = flatten_and_permute_latents(b)
    mu2d = mu.view(batch_size * z_dim)
    sigma2d = sigma.view(batch_size * z_dim)
    eta2d = eta.view(batch_size * z_dim)
    delta2d = delta.view(batch_size * z_dim)
    x2d, y2d = cheng_sample_1d(z2d, b2d, mu2d, sigma2d, eta2d, delta2d, k)

    x = unflatten_and_unpermute_latents(x2d, batch_size)
    y = unflatten_and_unpermute_latents(y2d, batch_size)

    return x, y


def cheng_antithetic_1d(z, b, x, mu, sigma, k):
    r"""Given a set of samples x1, ..., xk \sim N(mu, sigma^2),
    we can compute sample moments, eta and delta. Then we estimate
    the approximate antithetic sample moments (eta_anti, delta_anti).
    Critically, eta_anti and delta_anti are RV. We can use Cheng
    to generate samples y1, ..., yk to match antithetic moments.

    @param z: torch.Tensor
              shape: batch_size x k - 1
              samples from unit Gaussian
    @param b: torch.Tensor
              shape: batch_size x k - 1
              samples from Bernoulli
    @param x: torch.Tensor
              shape: batch_size x k
              samples drawn from N(mu, sigma)
    @param mu: torch.Tensor
               shape: batch_size
               population mean
    @param sigma: torch.Tensor
                  shape: batch_size
                  popuation standard deviation
    @param k: integer
              number of samples
    @return x_anti: torch.Tensor
                    shape: batch_size x k
                    antithetic samples with x ~ N(mu, sigma)
    @return y_anti: torch.Tensor
                    shape: batch_size x k
                    samples prior to Helmert (for sanity checks)
    """
    assert z.size(1) == (k - 1)
    assert b.size(1) == (k - 1)
    assert x.size(1) == k

    eta = torch.mean(x, dim=1)
    delta = torch.std(x, dim=1)

    # antithetic mean (flip over population mean)
    eta_anti = 2 * mu - eta

    # https://onlinecourses.science.psu.edu/stat414/node/174/
    # S \sim chi^2_{k-1}
    S = (k - 1) * torch.pow(delta, 2) / torch.pow(sigma, 2)

    # 4th-power approximation by Hawkins and Wixley to avoid any negative sign issues.
    # S_anti = antithetic_wilson_hilferty(S, k - 1)
    S_anti = antithetic_hawkins_wixley(S, k - 1)

    delta_squared = torch.pow(sigma, 2) * S_anti / (k - 1.)
    delta_anti = torch.sqrt(delta_squared + 1e-15)

    x_anti, y_anti = cheng_sample_1d(z, b, mu, sigma, eta_anti, delta_anti, k)

    return x_anti, y_anti


def cheng_antithetic_nd(z, b, x, mu, sigma, k):
    r"""Treat n dimensions as independent samples. We flatten
    each of these inputs to be n*k. See <cheng_antithetic_1d> for
    more details.

    Shape of z and b tensors
        batch_size x (k - 1) x z_dim

    Shape of x tensor
        batch_size x k x z_dim

    Shape of mu, sigma, eta, delta
        batch_size x z_dim

    The returned samples will be of the size:
        batch_size x k x z_dim
    """
    assert z.size(1) == (k - 1)
    assert b.size(1) == (k - 1)
    assert x.size(1) == k

    batch_size, _, z_dim = z.size()
    z2d = flatten_and_permute_latents(z)
    b2d = flatten_and_permute_latents(b)
    x2d = flatten_and_permute_latents(x)
    mu2d = mu.view(batch_size * z_dim)
    sigma2d = sigma.view(batch_size * z_dim)

    x2d_anti, y2d_anti = cheng_antithetic_1d(z2d, b2d, x2d, mu2d, sigma2d, k)
    x_anti = unflatten_and_unpermute_latents(x2d_anti, batch_size)
    y_anti = unflatten_and_unpermute_latents(y2d_anti, batch_size)

    return x_anti, y_anti


def antithetic_wilson_hilferty(X1, dof):
    # Wilson, Edwin B., and Margaret M. Hilferty. "The distribution
    # of chi-square." Proceedings of the National Academy of
    # Sciences 17.12 (1931): 684-688.
    return dof * torch.pow(2. * (1. - (2. / (9. * dof))) - \
                           torch.pow(X1 / float(dof) + 1e-15, 1./3), 3)


def antithetic_hawkins_wixley(X1, dof):
    # Hawkins, D.M., Wixley, R.A.J., 1986. A note on the transformation
    # of chi-squared variables to normality. Amer. Statist. 40, 296-298.
    return dof * torch.pow(2 * (1. - (3./(16.*dof)) - (7./(512.*dof**2)) + (231./(8192.*dof**3))) - \
                           torch.pow(X1 / float(dof) + 1e-15, 1./4), 4)


def unsqueeze_and_repeat(x, n, dim=1):
    x = x.unsqueeze(dim=dim)
    n_dims = x.dim()
    sizes = [1 for i in xrange(n_dims)]
    sizes[dim] = n
    x = x.repeat(*sizes)
    return x
