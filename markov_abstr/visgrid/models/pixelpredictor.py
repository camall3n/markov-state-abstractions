from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from .autoencoder import AutoEncoder
from .simplenet import SimpleNet
from .nnutils import one_hot

class PixelPredictor(AutoEncoder):
    def __init__(self,
                 n_actions,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=0.001,
                 coefs=None):
        super().__init__(n_actions, input_shape, n_latent_dims, n_hidden_layers, n_units_per_layer,
                         lr, coefs)
        self.transition_model = SimpleNet(
            n_inputs=(n_latent_dims + n_actions),
            n_outputs=n_latent_dims,
            n_units_per_layer=n_units_per_layer,
            n_hidden_layers=n_hidden_layers,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def compute_loss(self, x0, a, x1):
        z0 = self.encode(x0)
        transition_input = torch.cat((z0, one_hot(a, self.n_actions)), dim=-1)
        z1 = self.transition_model(transition_input)
        loss = self.mse(x1, self.decode(z1))
        return loss

    def train_batch(self, x0, x1, a, *args, **kwargs):
        self.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(x0, a, x1)
        loss.backward()
        self.optimizer.step()
        return loss
