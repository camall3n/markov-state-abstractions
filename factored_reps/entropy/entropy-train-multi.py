import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import torch

from markov_abstr.visgrid.models.phinet import PhiNet

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

n_dims = 3

sigma = 1.0e-2
sigma_tch = torch.as_tensor(sigma)
eps = 1.0e-5
eps_tch = torch.as_tensor(eps)

def compute_entropy_np(x):
    dx = (x[np.newaxis] - x[:, np.newaxis])
    K = lambda x: np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return np.log(len(x) -
                  1) - 1 / len(x) * np.sum(np.log(eps + np.sum(K(dx), axis=1) - K(0)), axis=0)

def compute_entropy_torch(x):
    dx = (x.unsqueeze(0) - x.unsqueeze(1))
    K = lambda x: torch.exp(-x**2 /
                            (2 * sigma_tch**2)) / torch.as_tensor(sigma_tch * np.sqrt(2 * np.pi),
                                                                  dtype=torch.float32)
    return torch.log(torch.tensor(x.shape[0] - 1, dtype=torch.float32)) - 1 / len(x) * torch.sum(
        torch.log(eps + torch.sum(K(dx), dim=1) - K(torch.tensor(0))), dim=0)

N = 1000
x = [np.random.normal(0, (i + 1), N) for i in range(n_dims)]
h = [np.log(eps + (i + 1) * np.sqrt(2 * np.pi * np.e)) for i in range(n_dims)]
x_np = np.stack(x, axis=1)
x_tch = torch.as_tensor(x_np, dtype=torch.float32)
print(compute_entropy_np(x_np))
print(compute_entropy_torch(x_tch))
print(h)

#%% -----

class EntropyNet(PhiNet):
    def __init__(self,
                 input_shape=1,
                 n_latent_dims=1,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=1e-3):
        super().__init__(input_shape=input_shape,
                         n_latent_dims=n_latent_dims,
                         n_hidden_layers=n_hidden_layers,
                         n_units_per_layer=n_units_per_layer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def train_batch(self, x, ascend=False):
        self.train()
        self.optimizer.zero_grad()
        z = self.phi(x)
        loss = torch.mean(compute_entropy_torch(z))
        if ascend:
            loss = -loss
        loss.backward()
        self.optimizer.step()
        return loss

net = EntropyNet(input_shape=n_dims,
                 n_latent_dims=n_dims,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=1e-3)
net.phi(x_tch).mean(dim=0)

#%%
n_updates = 400

losses = []
reps = []
for i in range(n_updates):
    loss = net.train_batch(x_tch)
    losses.append(loss)
    with torch.no_grad():
        rep = net.phi(x_tch)
        reps.append(rep.squeeze())

for i in range(n_updates):
    loss = net.train_batch(x_tch, ascend=True)
    losses.append(-loss)
    with torch.no_grad():
        rep = net.phi(x_tch)
        reps.append(rep.squeeze())
reps = torch.stack(reps).numpy()
losses = torch.stack(losses).detach().numpy()

#%%

fig = plt.figure(figsize=(10, 4))
plt.plot(range(2 * n_updates), losses)
plt.ylabel('entropy')
plt.xlabel('n_updates')
plt.vlines(n_updates, min(losses), max(losses), linestyles='dashed')
plt.tight_layout()

fig = plt.figure(figsize=(10, 4))
axes = fig.subplots(nrows=2, ncols=5, sharey=True, sharex=True)
axes = axes.flatten()
idx = np.arange(0, 2 * n_updates, 2 * n_updates // 10)

for i in range(len(axes)):
    for dim in range(n_dims):
        sns.distplot(reps[idx[i]][:, dim], hist=False, ax=axes[i], label=str(dim))
    if i >= len(axes) / 2:
        axes[i].set_xlabel('z')
    axes[i].set_title(str(idx[i]) + ' updates')
    axes[i].legend()
axes[0].set_ylabel('p(z)')
axes[5].set_ylabel('p(z)')

plt.tight_layout()
plt.show()
