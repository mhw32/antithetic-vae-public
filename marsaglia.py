from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np

import torch
from torch.autograd import Variable

from utils import (
    flatten_and_permute_latents,
    unflatten_and_unpermute_latents,
)
from antithetic import antithetic_hawkins_wixley


def marsaglia_antithetic_1d(z, x, mu, sigma, k):
    assert z.size(1) == (k - 1)
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

    x_anti = marsaglia_sample_1d(z, mu, sigma, eta_anti, delta_anti, k)

    return x_anti


def marsaglia_antithetic_nd(z, x, mu, sigma, k):
    assert z.size(1) == (k - 1)
    assert x.size(1) == k

    batch_size, _, z_dim = z.size()
    z2d = flatten_and_permute_latents(z)
    x2d = flatten_and_permute_latents(x)
    mu2d = mu.view(batch_size * z_dim)
    sigma2d = sigma.view(batch_size * z_dim)

    x2d_anti = marsaglia_antithetic_1d(z2d, x2d, mu2d, sigma2d, k)
    x_anti = unflatten_and_unpermute_latents(x2d_anti, batch_size)

    return x_anti


def marsaglia_sample_1d(z, mu, sigma, eta, delta, k):
    r"""Similar to Pullin but a simpler algorithm
    and it requires fewer random choices.
    """
    assert z.size(1) == (k - 1)
    bsize = z.size(0)
    a = eta
    r = delta * math.sqrt(k - 1)

    s = torch.sum(torch.pow(z, 2), dim=1)
    i = Variable(torch.arange(1, k).unsqueeze(0).repeat(bsize, 1))
    if mu.is_cuda:
        i = i.cuda()
    z = z * torch.pow((k - i) * (k - i + 1) * s.unsqueeze(1), -0.5)
    # slow form below: vectorized form above!
    # for _i in xrange(k - 1):
    #     i = _i + 1
    #     z[:, _i] = z[:, _i] * torch.pow((k-i)*(k-i+1)*s, -0.5)

    def get_t(i):
        return torch.sum(z[:, 0:i], dim=1)

    x_1 = (1 - k) * r * z[:, 0] + a
    x_k = r * get_t(k - 1) + a

    x = []
    for _i in range(1, k - 1):
        i = _i + 1
        x_i = (get_t(i - 1) + (i - k) * z[:, _i]) * r + a
        x.append(x_i)

    x = [x_1] + x + [x_k]
    x = torch.stack(x).t().contiguous()

    return x


def marsaglia_sample_nd(z, mu, sigma, eta, delta, k):
    assert z.size(1) == (k - 1)

    batch_size, _, z_dim = z.size()
    z2d = flatten_and_permute_latents(z)
    mu2d = mu.view(batch_size * z_dim)
    sigma2d = sigma.view(batch_size * z_dim)
    eta2d = eta.view(batch_size * z_dim)
    delta2d = delta.view(batch_size * z_dim)
    x2d = marsaglia_sample_1d(z2d, mu2d, sigma2d, eta2d, delta2d, k)
    x = unflatten_and_unpermute_latents(x2d, batch_size)

    return x
