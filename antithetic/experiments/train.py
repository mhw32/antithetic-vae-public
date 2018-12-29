from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import math
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from .datasets import build_dataset
from .models import build_model
from .elbo import build_objective
from .utils import AverageMeter
from . import DSET_DOMAIN, REPO_ROOT, DSET_TO_DIST, DIST_DOMAIN, DSET_TO_SIZE


if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='StaticMNIST|DynamicMNIST|FashionMNIST|Omniglot|Histopathology|Caltech101|FreyFaces')
    parser.add_argument('model', type=str, help='vanilla|cheng|marsaglia')
    parser.add_argument('--backprop', action='store_true', default=False,
                        help='backprop through Cheng/Marsaglia transform [default: False]')
    parser.add_argument('--z-dim', type=int, default=40, metavar='N',
                        help='number of latent dimensions [default: 40]')
    parser.add_argument('--n-samples', type=int, default=8, metavar='N',
                        help='number of samples to use in IWAE [default: 8]')
    parser.add_argument('--iwae', action='store_true', default=False,
                        help='if True, train with IWAE loss [default: False]')
    parser.add_argument('--n-norm-flows', type=int, default=False,
                        help='number of normalizing flows [default: 0]')
    parser.add_argument('--n-volume-flows', type=int, default=False,
                        help='number of volume preserving flows [default: 0]')
    parser.add_argument('--hidden-dim', type=int, default=300,
                        help='number of hidden units in VAE [default: 300]')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 128]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate [default: 3e-4]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--out-dir', type=str, default='./trained_models',
                        help='where to save outputs [default: ./trained_models]')
    parser.add_argument('--seed', type=int, default=1, help='random seed [default: 1]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='cast CUDA on variables if available [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # command line checks
    assert args.dataset in DSET_DOMAIN, \
        "dataset <%s> not recognized." % args.dataset
    args.data_dist = DSET_TO_DIST[args.dataset]
    args.bernoulli_data = args.data_dist == 'bernoulli'
    assert args.data_dist in DIST_DOMAIN, \
        "distribution <%s> not recognized." % args.data_dist

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    train_loader = torch.utils.data.DataLoader(
        build_dataset(args.dataset, train=True),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        build_dataset(args.dataset, train=False),
        batch_size=args.batch_size, shuffle=True)

    model = build_model(
        DSET_TO_SIZE[args.dataset], args.z_dim, args.n_samples, hidden_dim=args.hidden_dim,
        n_norm_flows=args.n_norm_flows, n_volume_flows=args.n_volume_flows,
        data_dist=DSET_TO_DIST[args.dataset], backprop=args.backprop)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    use_flow = args.n_norm_flows > 0 or args.n_volume_flows > 0
    elbo_loss_fn = build_objective( iwae=args.iwae, use_flow=use_flow,
                                    bernoulli=args.bernoulli_data)


    def compute_elbo(model, data):
        r"""Helper function to hide all the branching into one place."""
        
        if args.n_norm_flows > 0:
            if args.bernoulli_data:
                recon_x_mu, zk, z0, z0_mu, z0_logvar, log_abs_det_jacobian = model(data)
                loss = elbo_loss_fn(recon_x_mu, data, zk, z0, z0_mu, z0_logvar,
                                    log_abs_det_jacobian=log_abs_det_jacobian)
            else:
                recon_x_mu, recon_x_logvar, zk, z0, z0_mu, z0_logvar, log_abs_det_jacobian = model(data)
                loss = elbo_loss_fn(recon_x_mu, recon_x_logvar, data, zk, z0, z0_mu, z0_logvar,
                                    log_abs_det_jacobian=log_abs_det_jacobian)

        elif args.n_volume_flows > 0:
            if args.bernoulli_data:
                recon_x_mu, zk, z0, z0_mu, z0_logvar = model(data)
                loss = elbo_loss_fn(recon_x_mu, data, zk, z0, z0_mu, z0_logvar,
                                    log_abs_det_jacobian=None)
            else:
                recon_x_mu, recon_x_logvar, zk, z0, z0_mu, z0_logvar = model(data)
                loss = elbo_loss_fn(recon_x_mu, recon_x_logvar, data, zk, z0, z0_mu, z0_logvar,
                                    log_abs_det_jacobian=None)

        else:
            if args.bernoulli_data:
                recon_x_mu, z, z_mu, z_logvar = model(data)
                loss = elbo_loss_fn(recon_x_mu, data, z, z_mu, z_logvar)
            else:
                recon_x_mu, recon_x_logvar, z, z_mu, z_logvar = model(data)
                loss = elbo_loss_fn(recon_x_mu, recon_x_logvar, data, z, z_mu, z_logvar)

            if args.track_stats:
                ziid = model._reparameterize(z_mu, z_logvar)
                return loss, ziid, z, z_mu, z_logvar

        return loss


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = len(data)
            data = data.to(device)
            data = data.view(-1, DSET_TO_SIZE[args.dataset])

            optimizer.zero_grad()

            loss = compute_elbo(model, data)
            loss_meter.update(loss.item(), batch_size)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tELBO: {:.6f}'.format(
                        epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        -loss_meter.avg))

        print('====> Train Epoch: {}\tELBO: {:.4f}'.format(epoch, -loss_meter.avg))
        return loss_meter.avg


    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            for data, _ in test_loader:
                batch_size = len(data)
                data = data.to(device)
                data = data.view(-1, DSET_TO_SIZE[args.dataset])

                loss = model.estimate_marginal(data, n_samples=100)
                loss_meter.update(loss.item(), batch_size)

                pbar.update()
            pbar.close()

        print('====> Test Epoch: {}\tlog p(x): {:.4f}'.format(epoch, -loss_meter.avg))
        return loss_meter.avg


    best_loss = sys.maxint
    track_train_elbo = np.zeros(args.epochs)
    track_test_marginal = np.zeros(args.epochs)

    for epoch in range(args.epochs):
        train_elbo = train(epoch)
        test_marginal = test(epoch)
        track_train_elbo[epoch] = train_elbo
        track_test_marginal[epoch] = test_marginal

        np.save(os.path.join(args.out_dir, 'train_elbo.npy'), track_train_elbo)
        np.save(os.path.join(args.out_dir, 'test_marginal.npy'), track_test_marginal)
