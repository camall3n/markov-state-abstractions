import numpy as np
import torch
import torch.nn

from markov_abstr.visgrid.models.nnutils import Network, one_hot, extract
from markov_abstr.visgrid.models.qnet import QNet

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
