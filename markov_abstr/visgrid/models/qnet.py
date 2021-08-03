import numpy as np
import torch
import torch.nn

from .nnutils import Network, one_hot, extract

class QNet(Network):
    def __init__(self, n_features, n_actions, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(n_features, n_actions)])
        else:
            self.layers.extend([torch.nn.Linear(n_features, n_units_per_layer), torch.nn.ReLU()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.ReLU()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_actions)])

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, z):
        return self.model(z)
