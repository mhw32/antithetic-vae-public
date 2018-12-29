from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

REPO_ROOT = os.path.realpath(os.path.dirname(__file__))

DSET_DOMAIN = [
    'StaticMNIST',
    'DynamicMNIST',
    'FashionMNIST',
    'Omniglot',
    'Histopathology',
    'FreyFaces',
    'Caltech101',
]

DIST_DOMAIN = ['bernoulli', 'logistic']

DSET_TO_DIST = {
    'StaticMNIST': 'bernoulli',
    'DynamicMNIST': 'bernoulli',
    'FashionMNIST': 'logistic',
    'Omniglot': 'bernoulli',
    'Histopathology': 'logistic',
    'FreyFaces': 'logistic',
    'Caltech101': 'bernoulli',
}

DSET_TO_SIZE = {
    'StaticMNIST': 28*28,
    'DynamicMNIST': 28*28,
    'IndexMNIST': 28*28,
    'FashionMNIST': 28*28,
    'Omniglot': 28*28,
    'Histopathology': 28*28,
    'FreyFaces': 28*20,
    'Caltech101': 28*28,
}
