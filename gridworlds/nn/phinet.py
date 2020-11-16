import numpy as np
import torch
import torch.nn

from .nnutils import Network, Reshape

class PhiNet(Network):
    def __init__(self,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 final_activation=torch.nn.Tanh):
        super().__init__()
        self.input_shape = input_shape

        shape_flat = np.prod(self.input_shape)

        self.layers = []
        self.layers.extend([Reshape(-1, shape_flat)])
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(shape_flat, n_latent_dims)])
        else:
            self.layers.extend([torch.nn.Linear(shape_flat, n_units_per_layer), torch.nn.Tanh()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.Tanh()] * (n_hidden_layers - 1))
            self.layers.extend([
                torch.nn.Linear(n_units_per_layer, n_latent_dims),
            ])
        if final_activation is not None:
            self.layers.extend([final_activation()])
        self.phi = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        z = self.phi(x)
        return z
