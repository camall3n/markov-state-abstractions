import math
import numpy as np
import torch
import torch.nn

from .nnutils import Network
from .phinet import PhiNet
from .invnet import InvNet
from .fwdnet import FwdNet
from .cpcnet import CPCNet

class FeatureNet(Network):
    def __init__(self, n_actions, input_shape=2, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32, lr=0.001, inv_steps_per_fwd=5):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.inv_steps_per_fwd = inv_steps_per_fwd

        self.phi = PhiNet(input_shape=input_shape, n_latent_dims=n_latent_dims, n_units_per_layer=n_units_per_layer, n_hidden_layers=n_hidden_layers, lr=lr)

        self.fwd_model = FwdNet(n_actions=n_actions, n_latent_dims=n_latent_dims, n_hidden_layers=n_hidden_layers, n_units_per_layer=n_units_per_layer, lr=lr)

        self.inv_model = InvNet(n_actions=n_actions, n_latent_dims=n_latent_dims, n_units_per_layer=n_units_per_layer, n_hidden_layers=n_hidden_layers, lr=lr)

        self.cpc_model = CPCNet(n_latent_dims=n_latent_dims, n_hidden_layers=1, n_units_per_layer=n_units_per_layer)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_inv_loss(self, a_logits, a):
        return self.cross_entropy(input=a_logits, target=a)

    def compute_fwd_loss(self, z0, z1, z1_hat):
        eps = 1e-6
        error = torch.sqrt(torch.sum(torch.pow(z1_hat - z1, 2), dim=-1))
        dz = torch.sqrt(torch.sum(torch.pow(z1 - z0, 2), dim=-1))
        return torch.mean(error / (dz + eps))
        # return self.mse(z1,z1_hat)

    def compute_cpc_loss(self, z1, z1_hat):
        N = len(z1)
        # shuffle next states
        idx = torch.randperm(N)
        z1_alt = z1.view(N,-1)[idx].view(z1.size())

        # concatenate positive and negative examples
        c = torch.cat([z1_hat, z1_hat], dim=0)
        z = torch.cat([z1, z1_alt], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.cpc_model(c, z)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def compute_diversity_loss(self, z0, z1):
        # Encourage the largest difference to be non-zero
        return 0.1*torch.mean(torch.exp(-100 * torch.max(torch.pow(z0-z1,2),dim=-1)[0]))

    def kde_entropy(self, x, sigma=1.0e-2, eps=1.0e-5):
        sigma = torch.as_tensor(sigma)
        eps = torch.as_tensor(eps)
        # pairwise differences
        dx = (x.unsqueeze(0) - x.unsqueeze(1))
        K = lambda x: torch.exp(-x**2 / (2*sigma**2)) / torch.as_tensor(sigma * np.sqrt(2*np.pi), dtype=torch.float32)
        return torch.log(torch.tensor(x.shape[0]-1, dtype=torch.float32)) - 1/len(x) * torch.sum(torch.log(eps + torch.sum(K(dx), dim=1)-K(torch.tensor(0.0))), dim=0)

    def compute_entropy_loss(self, z0, z1, a):
        entropies = []
        for a_idx in range(self.n_actions):
            mask = (a==a_idx)
            if torch.any(mask):
                dz_a = torch.masked_select((z1-z0), torch.stack([mask,mask],dim=-1)).reshape(-1,self.n_latent_dims)
                h_a = self.kde_entropy(dz_a)
                entropies.append(h_a)
        loss = torch.mean(torch.stack(entropies))
        return loss

    def compute_factored_loss(self, z0, z1):
        eps = 1e-6
        dz = z1-z0
        l1 = torch.sum(torch.abs(dz), dim=-1)
        lmax = torch.max(torch.abs(dz), dim=-1)[0]
        return torch.mean(l1 / lmax)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, z0, z1):
        a_logits = self.inv_model(z0, z1)
        return torch.argmax(a_logits, dim=-1)

    def train_batch(self, x0, x1, a, model='inv'):
        self.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        z1_hat = self.fwd_model(z0, a)
        loss = 0
        if model in ['inv', 'all']:
            a_hat = self.inv_model(z0, z1)
            loss += self.compute_inv_loss(a_logits=a_hat, a=a)
        if model in ['fwd', 'all']:
            loss += 0.1 * self.compute_fwd_loss(z0, z1, z1_hat)
        if model in ['cpc', 'all']:
            # loss += self.compute_distinguish_loss(x0, a, x1)
            loss += self.compute_cpc_loss(z1, z1_hat)
        if model in ['factor', 'all']:
            loss += 0.1 * self.compute_factored_loss(z0, z1)
        # if model in ['entropy', 'all']:
        #     loss += .2 * self.compute_entropy_loss(z0, z1, a)
        loss.backward()
        self.optimizer.step()
        return loss
