import numpy as np
import torch
import torch.nn

from .nnutils import Network

class SimpleNet(Network):
    def __init__(self, n_inputs, n_outputs, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_outputs = n_outputs
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(n_inputs, n_outputs)])
        else:
            self.layers.extend([torch.nn.Linear(n_inputs, n_units_per_layer), torch.nn.Tanh()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.Tanh()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_outputs)])

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, z0):
        a_logits = self.model(z0)
        return a_logits
