from collections import defaultdict
import numpy as np
import torch
import torch.nn

from .nnutils import Network
from .phinet import PhiNet
from .invnet import InvNet
from .fwdnet import FwdNet
from .contrastivenet import ContrastiveNet
from .invdiscriminator import InvDiscriminator

class FeatureNet(Network):
    def __init__(self,
                 n_actions,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=0.001,
                 coefs=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.coefs = defaultdict(lambda: 1.0)
        if coefs is not None:
            for k, v in coefs.items():
                self.coefs[k] = v

        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=n_latent_dims,
                          n_units_per_layer=n_units_per_layer,
                          n_hidden_layers=n_hidden_layers)
        # self.fwd_model = FwdNet(n_actions=n_actions, n_latent_dims=n_latent_dims, n_hidden_layers=n_hidden_layers, n_units_per_layer=n_units_per_layer)
        self.inv_model = InvNet(n_actions=n_actions,
                                n_latent_dims=n_latent_dims,
                                n_units_per_layer=n_units_per_layer,
                                n_hidden_layers=n_hidden_layers)
        self.inv_discriminator = InvDiscriminator(n_actions=n_actions,
                                                  n_latent_dims=n_latent_dims,
                                                  n_units_per_layer=n_units_per_layer,
                                                  n_hidden_layers=n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=n_latent_dims,
                                            n_hidden_layers=1,
                                            n_units_per_layer=n_units_per_layer)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def inverse_loss(self, z0, z1, a):
        if self.coefs['L_inv'] == 0.0:
            return torch.tensor(0.0)
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def contrastive_inverse_loss(self, z0, z1, a):
        if self.coefs['L_coinv'] == 0.0:
            return torch.tensor(0.0)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)

        a_neg = torch.randint_like(a, low=0, high=self.n_actions)

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_extended = torch.cat([z1, z1], dim=0)
        a_pos_neg = torch.cat([a, a_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.inv_discriminator(z0_extended, z1_extended, a_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def ratio_loss(self, z0, z1):
        if self.coefs['L_rat'] == 0.0:
            return torch.tensor(0.0)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.discriminator(z0_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def distance_loss(self, z0, z1):
        if self.coefs['L_dis'] == 0.0:
            return torch.tensor(0.0)
        dz = torch.norm(z1 - z0, dim=-1, p=2)
        with torch.no_grad():
            max_dz = 0.1
        excess = torch.nn.functional.relu(dz - max_dz)
        return self.mse(excess, torch.zeros_like(excess))

    def oracle_loss(self, z0, z1, d):
        if self.coefs['L_ora'] == 0.0:
            return torch.tensor(0.0)

        dz = torch.cat(
            [torch.norm(z1 - z0, dim=-1, p=2),
             torch.norm(z1.flip(0) - z0, dim=-1, p=2)], dim=0)

        with torch.no_grad():
            counts = 1 + torch.histc(d, bins=36, min=0, max=35)
            inverse_counts = counts.sum() / counts
            weights = inverse_counts[d.long()]
            weights = weights / weights.sum()

        loss = self.mse(dz, d / 10.0)
        # loss += torch.sum(weights * (dz - d / 20.0)**2) # weighted MSE
        # loss = -torch.nn.functional.cosine_similarity(dz, d, 0)
        return loss

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, z0, z1):
        raise NotImplementedError
        # a_logits = self.inv_model(z0, z1)
        # return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, z0, z1, a, d):
        loss = 0
        loss += self.coefs['L_coinv'] * self.contrastive_inverse_loss(z0, z1, a)
        loss += self.coefs['L_inv'] * self.inverse_loss(z0, z1, a)
        # loss += self.coefs['L_fwd'] * self.compute_fwd_loss(z0, z1, z1_hat)
        loss += self.coefs['L_rat'] * self.ratio_loss(z0, z1)
        loss += self.coefs['L_dis'] * self.distance_loss(z0, z1)
        loss += self.coefs['L_ora'] * self.oracle_loss(z0, z1, d)
        return loss

    def train_batch(self, x0, x1, a, d):
        self.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        # z1_hat = self.fwd_model(z0, a)
        loss = self.compute_loss(z0, z1, a, d)
        loss.backward()
        self.optimizer.step()
        return loss
