from collections import defaultdict
import numpy as np
import torch
import torch.nn

from .nnutils import Network
from .phinet import PhiNet
from .invnet import InvNet
from .fwdnet import FwdNet
from .contrastivenet import ContrastiveNet

class FeatureNet(Network):
    def __init__(self, n_actions, input_shape=2, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32, lr=0.001, coefs=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.coefs = defaultdict(lambda:1.0)
        if coefs is not None:
            for k, v in coefs.items():
                self.coefs[k] = v

        self.phi = PhiNet(input_shape=input_shape, n_latent_dims=n_latent_dims, n_units_per_layer=n_units_per_layer, n_hidden_layers=n_hidden_layers)
        # self.fwd_model = FwdNet(n_actions=n_actions, n_latent_dims=n_latent_dims, n_hidden_layers=n_hidden_layers, n_units_per_layer=n_units_per_layer)
        self.inv_model = InvNet(n_actions=n_actions, n_latent_dims=n_latent_dims, n_units_per_layer=n_units_per_layer, n_hidden_layers=n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=n_latent_dims, n_hidden_layers=1, n_units_per_layer=n_units_per_layer)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def inverse_loss(self, z0, z1, a):
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def ratio_loss(self, z0, z1):
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)
        z1_neg = z1.view(N,-1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.discriminator(z0_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def distance_loss(self, z0, z1, i):
        if self.coefs['L_dis'] == 0.0:
            return torch.tensor(0.0)
        dz = torch.norm(z0[:, None] - z1, dim=-1, p=2)
        with torch.no_grad():
            idx = i.float()
            max_dz = torch.abs(idx.unsqueeze(-1)-(idx+1).unsqueeze(-2))/50
        excess = torch.nn.functional.relu(dz - max_dz)
        return self.mse(excess, torch.tensor(0.))

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, z0, z1):
        a_logits = self.inv_model(z0, z1)
        return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, z0, z1, a, i, model='all'):
        loss = 0
        if model in ['L_inv', 'all']:
            loss += self.coefs['L_inv'] * self.inverse_loss(z0, z1, a)
        # if model in ['L_fwd', 'all']:
        #     loss += self.coefs['L_fwd'] * self.compute_fwd_loss(z0, z1, z1_hat)
        if model in ['L_rat', 'all']:
            loss += self.coefs['L_rat'] * self.ratio_loss(z0, z1)
        if model in ['L_dis', 'all']:
            if self.coefs['L_dis'] > 0.0:
                loss += self.coefs['L_dis'] * self.distance_loss(z0, z1, i)
        return loss

    def train_batch(self, x0, x1, a, i, model='inv'):
        self.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        # z1_hat = self.fwd_model(z0, a)
        loss = self.compute_loss(z0, z1, a, i, model=model)
        loss.backward()
        self.optimizer.step()
        return loss
