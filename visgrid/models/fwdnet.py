import numpy as np
import torch
import torch.nn

from .nnutils import Network, one_hot

class FwdNet(Network):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions
        self.frozen = False

        self.fwd_layers = []
        if n_hidden_layers == 0:
            self.fwd_layers.extend([torch.nn.Linear(n_latent_dims+self.n_actions, n_latent_dims)])
        else:
            self.fwd_layers.extend([torch.nn.Linear(n_latent_dims + self.n_actions, n_units_per_layer), torch.nn.Tanh()])
            self.fwd_layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()] * (n_hidden_layers-1))
            self.fwd_layers.extend([torch.nn.Linear(n_units_per_layer, n_latent_dims)])
        # self.fwd_layers.extend([torch.nn.BatchNorm1d(n_latent_dims, affine=False)])
        self.fwd_model = torch.nn.Sequential(*self.fwd_layers)

    def forward(self, z, a):
        a_onehot = one_hot(a, depth=self.n_actions)
        context = torch.cat((z, a_onehot), -1)
        z_hat = self.fwd_model(context)
        return z_hat
