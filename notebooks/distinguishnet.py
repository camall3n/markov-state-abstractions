import numpy as np
import torch
import torch.nn

from .nnutils import Network, one_hot

class DistinguishNet(Network):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2*n_latent_dims+self.n_actions, 1)])
        else:
            self.layers.extend([torch.nn.Linear(2*n_latent_dims+self.n_actions, n_units_per_layer), torch.nn.Tanh()])
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()] * (n_hidden_layers-1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])
        self.layers.extend([torch.nn.Sigmoid()])
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, z0, a, z1):
        a_onehot = one_hot(a, depth=self.n_actions)
        context = torch.cat((z0, a_onehot, z1), -1)
        fakes = self.model(context).squeeze()
        return fakes
