r"""A collection of datasets that we will be using for training."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cPickle
from tqdm import tqdm
from scipy.io import loadmat

import torch
import numpy as np
from PIL import Image
from os.path import join
import torch.utils.data as data
from torchvision import datasets, transforms

from config import DSET_DOMAIN, REPO_ROOT


def build_dataset(dataset_name, train=True):
    r"""Helper function to load a MNIST-variant dataset.

    @param dataset_name: string
                         name of MNIST dataset.
                         StaticMNIST|DynamicMNIST|FashionMNIST|Omniglot
                         Histopathology|Caltech101|FreyFaces
    @param train: boolean [default: True]
                  load trin or test partition of dataset.
    """
    assert dataset_name in DSET_DOMAIN, \
        "dataset <%s> not recognized." % dataset_name

    # a few datasets need to dump things... use this folder
    data_dir = os.path.join(REPO_ROOT, 'datasets')
    data_dir = os.path.realpath(data_dir)
    data_dir = os.path.join(data_dir, dataset_name)

    if dataset_name == 'StaticMNIST':
        split = 'train' if train else 'test'
        return StaticMNIST(split=split)

    elif dataset_name == 'DynamicMNIST':
        if train:
            return datasets.MNIST(data_dir, train=train, download=True,
                                  transform=dynamic_binarize)
        else:
            # do not dynamically binarize the test set
            # but dont just take the rounded version (call bernoulli once)
            return load_dynamic_mnist_test_set(data_dir)

    elif dataset_name == 'FashionMNIST':
        return datasets.FashionMNIST(data_dir, train=train, download=True,
                                     transform=transforms.ToTensor())

    elif dataset_name == 'Omniglot':
        if train:
            return Omniglot(train=True, transform=torch.bernoulli)
        else:
            # do not dynamically binarize the test set
            # but dont just take the rounded version (call bernoulli once)
            return load_omniglot_test_set()

    elif dataset_name == 'Histopathology':
        split = 'training' if train else 'test'
        return Histopathology(split=split)

    elif dataset_name == 'Caltech101':
        # already binarized
        return Caltech101(train=train)

    elif dataset_name == 'FreyFaces':
        return FreyFaces(train=train)


def dynamic_binarize(x):
    # Dynamic Binarization
    x = transforms.ToTensor()(x)
    x = torch.bernoulli(x)
    return x


class StaticMNIST(data.Dataset):
    r"""Previously binarized MNIST.

    @param split: string [default: train]
                  train|val|test
    """
    train_data = np.load(os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_train.npy')))
    val_data   = np.load(os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_valid.npy')))
    test_data  = np.load(os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_test.npy')))

    def __init__(self, split='train'):
        super(StaticMNIST, self).__init__()
        self.split = split
        if self.split == 'train':
            self.data = self.train_data
        elif self.split == 'val':
            self.data = self.val_data
        elif self.split == 'test':
            self.data = self.test_data
        else:
            raise Exception('<%s> split not recognized.' % self.split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = torch.from_numpy(image).float()
        return image, index


def prep_static_mnist():
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_train.amat'))) as fp:
        lines = fp.readlines()
        data_train = lines_to_np_array(lines).astype('float32')
        np.save(os.path.realpath(
            os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_train.npy')),
            data_train)

    with open(os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_valid.amat'))) as fp:
        lines = fp.readlines()
        data_valid = lines_to_np_array(lines).astype('float32')
        np.save(os.path.realpath(
            os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_valid.npy')),
            data_valid)

    with open(os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_test.amat'))) as fp:
        lines = fp.readlines()
        data_test = lines_to_np_array(lines).astype('float32')
        np.save(os.path.realpath(
            os.path.join(REPO_ROOT, 'datasets/static_mnist/binarized_mnist_test.npy')),
            data_test)


class Omniglot(data.Dataset):
    r"""Binarized characters from various vocabularies.

    @param train: boolean [default: True]
                  use training split
    """
    mat_path = os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/omniglot/chardata.mat'))

    def __init__(self, train=True, transform=None):
        super(Omniglot, self).__init__()
        self.train = train
        self.transform = transform

        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

        self.data = loadmat(self.mat_path)
        if self.train:
            self.data = reshape_data(self.data['data'].T.astype('float32'))
        else:
            self.data = reshape_data(self.data['testdata'].T.astype('float32'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        return image, index


class Caltech101(data.Dataset):
    r"""Silhouettes of objects from various classes.

    @param train: boolean [default: True]
                  use training split
    """
    mat_path = os.path.realpath(
        os.path.join(
            REPO_ROOT,
            'datasets/caltech101/caltech101_silhouettes_28_split1.mat'))

    def __init__(self, train=True, transform=None):
        super(Caltech101, self).__init__()
        self.train = train
        self.transform = transform

        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

        self.data = loadmat(self.mat_path)
        if self.train:
            self.data = 1. - reshape_data(self.data['train_data'].astype('float32'))
        else:
            self.data = 1. - reshape_data(self.data['test_data'].astype('float32'))

        # shuffle data:
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        return image, index


class FreyFaces(data.Dataset):
    r"""Gray faces with emotions.

    @param train: boolean [default: True]
                  use training split
    """
    pickle_path = os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/freyfaces/freyfaces.pkl'))

    def __init__(self, train=True, transform=None):
        super(FreyFaces, self).__init__()
        self.train = train
        self.transform = transform

        with open(self.pickle_path) as fp:
            self.data = cPickle.load(fp)

        # shuffle data:
        np.random.shuffle(self.data)

        if self.train:
            self.data = self.data[0:1765, :]
        else:
            self.data = self.data[1765:, :]

        self.data = self.data.reshape(-1, 28, 20)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = torch.from_numpy(image).float()

        if self.transform is not None:
            image = self.transform(image)

        return image, index


class Histopathology(data.Dataset):
    """Grayscale Histopathology Dataset.

    See https://arxiv.org/pdf/1611.09630.pdf

    @param split: string
                  training|validation|test
    """
    pickle_path = os.path.realpath(
        os.path.join(REPO_ROOT, 'datasets/histopathology/histopathology.pkl'))

    def __init__(self, split='training'):
        super(Histopathology, self).__init__()

        with open(self.pickle_path) as fp:
            self.data = cPickle.load(fp)

        assert split in self.data.keys(), "<split> not recognized."
        self.data = self.data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        image = np.clip(image, 1./512, 1.-1./512)
        image = torch.from_numpy(image).float()
        return image, index


def load_dynamic_mnist_test_set(data_dir):
    # initial load we can take advantage of the dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor()),
        batch_size=100, shuffle=True)

    # load it back into numpy tensors...
    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    y_test = np.array(test_loader.dataset.test_labels.float().numpy(), dtype=int)

    # binarize once!!! (we don't dynamically binarize this)
    np.random.seed(777)
    x_test = np.random.binomial(1, x_test)

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test)

    # pytorch data loader
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    return test_dataset


def load_omniglot_test_set():
    # initial load we can take advantage of the dataloader
    test_loader = torch.utils.data.DataLoader(
        Omniglot(train=False, transform=transforms.ToTensor()),
        batch_size=100, shuffle=True)

    x_test = test_loader.dataset.data
    # binarize once!!! (we don't dynamically binarize this)
    np.random.seed(777)
    x_test = np.random.binomial(1, x_test)
    x_test = torch.from_numpy(x_test).float()

    # pytorch data loader
    test_dataset = torch.utils.data.TensorDataset(x_test, x_test)

    return test_dataset
