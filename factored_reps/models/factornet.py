from collections import defaultdict
import numpy as np
import torch
import torch.nn

from markov_abstr.visgrid.models.nnutils import Network
from markov_abstr.visgrid.models.simplenet import SimpleNet

class FactorNet(Network):
    def __init__(self, n_latent_dims=2, lr=0.001, coefs=None):
        super().__init__()
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.coefs = defaultdict(lambda: 1.0)
        self.set_coefs(coefs)

        self.encoder = SimpleNet(n_inputs=n_latent_dims,
                                 n_outputs=n_latent_dims,
                                 n_hidden_layers=1,
                                 n_units_per_layer=32)
        self.decoder = SimpleNet(n_inputs=n_latent_dims,
                                 n_outputs=n_latent_dims,
                                 n_hidden_layers=1,
                                 n_units_per_layer=32)
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_coefs(self, coefs=None):
        if coefs is not None:
            for k, v in coefs.items():
                self.coefs[k] = v

    def forward(self, x):
        return self.encoder(x)

    def compute_factored_loss(self, z0, z1):
        eps = 1e-6
        dz = z1 - z0
        l1 = torch.sum(torch.abs(dz), dim=-1)
        lmax = torch.max(torch.abs(dz), dim=-1)[0]
        return torch.mean(l1 / lmax)

    def compute_loss(self, x0, x1, coefs=None):
        self.set_coefs(coefs)

        x0 = x0.detach()
        x1 = x1.detach()
        z0 = self.encoder(x0)
        z1 = self.encoder(x1)
        x0_hat = self.decoder(z0)
        x1_hat = self.decoder(z1)

        l_fac = self.compute_factored_loss(z0, z1)
        l_recons = (self.mse(x0, x0_hat) + self.mse(x1, x1_hat)) / 2.0
        loss = self.coefs['L_fac'] * l_fac + self.coefs['L_rec'] * l_recons
        return {'L': loss, 'L_fac': l_fac, 'L_rec': l_recons}

    def train_batch(self, x0, x1, coefs=None):
        self.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(x0, x1, coefs)
        loss['L'].backward()
        self.optimizer.step()
        return dict([(key, val.detach()) for key, val in loss.items()])
