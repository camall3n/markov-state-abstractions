from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from .nnutils import Network, Reshape
from .phinet import PhiNet

class AutoEncoder(Network):
    def __init__(self,
                 n_actions,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=0.001,
                 coefs=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.coefs = defaultdict(lambda: 1.0)
        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=n_latent_dims,
                          n_units_per_layer=n_units_per_layer,
                          n_hidden_layers=n_hidden_layers)
        self.reverse_phi = PhiNet(input_shape=input_shape,
                                  n_latent_dims=n_latent_dims,
                                  n_units_per_layer=n_units_per_layer,
                                  n_hidden_layers=n_hidden_layers)
        self.reverse_phi.phi = nn.Sequential(
            *reversed([Reshape(-1, *input_shape), nn.Tanh()] + [
                nn.Linear(l.out_features, l.in_features) if isinstance(l, nn.Linear) else l
                for l in self.reverse_phi.layers[1:-1]
            ]))
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, x0):
        return self.phi(x0)

    def decode(self, z0):
        return self.reverse_phi(z0)

    def compute_loss(self, x0):
        loss = self.mse(x0, self.decode(self.encode(x0)))
        return loss

    def train_batch(self, x0, *args, **kwargs):
        self.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(x0)
        loss.backward()
        self.optimizer.step()
        return loss
