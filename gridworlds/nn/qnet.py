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

class FactoredQNet(Network):
    def __init__(self, n_features, n_actions, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        self.q = torch.nn.ModuleList(
            [QNet(1, n_actions, n_hidden_layers, n_units_per_layer) for _ in range(n_features)])

    def forward(self, z, mask=None, reduce=True):
        if mask is not None:
            assert z.shape[-1] == mask.shape[-1]
            assert reduce, "'reduce' must be True when using 'mask'"
        else:
            mask = torch.ones(self.n_features)
        mask = mask.detach()
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        z = z.unsqueeze(-1)
        q = torch.stack([self.q[i](z[:, i]) for i in range(self.n_features)], dim=-1)
        masked_q = torch.matmul(q, mask)
        if reduce:
            return masked_q
        else:
            return q
