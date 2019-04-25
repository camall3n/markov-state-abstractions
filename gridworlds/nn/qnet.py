import numpy as np
import torch
import torch.nn

from .nnutils import Network

class QNet(Network):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(n_latent_dims, n_actions)])
        else:
            self.layers.extend([torch.nn.Linear(n_latent_dims, n_units_per_layer), torch.nn.ReLU()])
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.ReLU()] * (n_hidden_layers-1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_actions)])

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, z):
        a_logits = self.model(z)
        return a_logits
