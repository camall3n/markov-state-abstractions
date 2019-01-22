import numpy as np
import torch
import torch.nn

from .nnutils import Network

class InvNet(Network):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32, lr=0.001):
        super().__init__()
        self.n_actions = n_actions
        self.lr = lr
        self.frozen = False

        self.inv_layers = []
        self.inv_layers.extend([torch.nn.Linear(2 * n_latent_dims, n_units_per_layer), torch.nn.Tanh()])
        self.inv_layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.Tanh()] * (n_hidden_layers-1))
        self.inv_layers.extend([torch.nn.Linear(n_units_per_layer, self.n_actions)])
        self.inv_model = torch.nn.Sequential(*self.inv_layers)

    def forward(self, z0, z1):
        context = torch.cat((z0,z1), -1)
        a_logits = self.inv_model(context)
        return a_logits

    def freeze(self):
        if not self.frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.frozen = True

    def unfreeze(self):
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = True
            self.frozen = False
