import numpy as np
import torch
import torch.nn

from .nnutils import Network, one_hot

class CPCNet(Network):
    def __init__(self, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2*n_latent_dims, 1)])
        else:
            self.layers.extend([torch.nn.Linear(2*n_latent_dims, n_units_per_layer), torch.nn.Tanh()])
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()] * (n_hidden_layers-1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])
        self.layers.extend([torch.nn.Sigmoid()])
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, c, z):
        context = torch.cat((c, z), -1)
        fakes = self.model(context).squeeze()
        return fakes
