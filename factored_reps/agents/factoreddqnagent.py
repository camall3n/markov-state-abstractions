import numpy as np
import torch
import torch.nn.functional as F

from factored_reps.models.factoredqnet import FactoredQNet
from markov_abstr.visgrid.models.nnutils import one_hot, extract
from markov_abstr.visgrid.agents.dqnagent import DQNAgent, tch

class FactoredDQNAgent(DQNAgent):
    def __init__(self,
                 n_features,
                 n_actions,
                 phi,
                 lr=0.001,
                 epsilon=0.05,
                 batch_size=16,
                 train_phi=False,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 gamma=0.9,
                 factored=True):
        assert factored, 'FQN with dense QNet is not supported yet'
        super().__init__(n_features, n_actions, phi, lr, epsilon, batch_size, train_phi,
                         n_hidden_layers, n_units_per_layer, gamma, factored)
        self.make_qnet = FactoredQNet
        self.reset()

    def get_q_predictions(self, batch):
        z = self.phi(torch.stack(tch(batch.x, dtype=torch.float32)))
        if not self.train_phi:
            z = z.detach()
        a = torch.stack(tch(batch.a, dtype=torch.int64))
        qi_acted = extract(self.q(z, reduce=False), idx=a, idx_dim=-2)
        return qi_acted

    def get_q_targets(self, batch):
        with torch.no_grad():
            z = self.phi(torch.stack(tch(batch.x, dtype=torch.float32)))
            zp = self.phi(torch.stack(tch(batch.xp, dtype=torch.float32)))
            # Compute Double-Q targets
            ap = torch.argmax(self.q(zp), dim=-1)
            vp = extract(self.q_target(zp, reduce=False), idx=ap, idx_dim=1)
            not_done_idx = (1 - torch.stack(tch(batch.done, dtype=torch.float32)))
            not_done_idx = not_done_idx.view(-1, 1).expand_as(vp)
            r = torch.stack(tch(batch.r, dtype=torch.float32))
            r = r.view(-1, 1).expand_as(vp)
            qi_targets = (r + self.gamma * vp * not_done_idx) / self.n_features
        return qi_targets
