r"""Approximations for antithetic random variables."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch 


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
