import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from notebooks.phinet import PhiNet

sigma = 1.0e-2
sigma_tch = torch.as_tensor(sigma)
eps = 1.0e-5
eps_tch = torch.as_tensor(eps)

def compute_entropy_np(x):
    dx = (x[np.newaxis] - x[:, np.newaxis])
    K = lambda x: np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return np.log(len(x)-1) - 1/len(x) * np.sum(np.log(eps + np.sum(K(dx), axis=1)-K(0)), axis=0)

def compute_entropy_torch(x):
    dx = (x.unsqueeze(0) - x.unsqueeze(1))
    K = lambda x: torch.exp(-x**2 / (2*sigma_tch**2)) / torch.as_tensor(sigma_tch * np.sqrt(2*np.pi), dtype=torch.float32)
    return torch.log(torch.tensor(x.shape[0]-1, dtype=torch.float32)) - 1/len(x) * torch.sum(torch.log(eps + torch.sum(K(dx), dim=1)-K(torch.tensor(0))), dim=0)

N = 1000
x_np = np.random.uniform(0,1,N)
x_tch = torch.as_tensor(x_np, dtype=torch.float32)
print(compute_entropy_np(x_np))
print(compute_entropy_torch(x_tch))

class EntropyNet(PhiNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_batch(self, x):
        self.train()
        self.optimizer.zero_grad()
        z = self.phi(x)
        loss = compute_entropy_torch(z)
        loss.backward()
        self.optimizer.step()
        return loss

net = EntropyNet(input_shape=1, n_latent_dims=1, n_hidden_layers=1, n_units_per_layer=32, lr=1e-3)

n_updates = 50

losses = []
reps = []
for i in range(n_updates):
    loss = net.train_batch(x_tch)
    losses.append(loss)
    with torch.no_grad():
        rep = net.phi(x_tch)
        reps.append(rep.squeeze())
reps = torch.stack(reps).numpy()
fig = plt.figure(figsize=(10,4))
plt.plot(range(n_updates), losses)
plt.ylabel('entropy')
plt.xlabel('n_updates')


fig = plt.figure(figsize=(10,4))
axes = fig.subplots(nrows=1, ncols=5, sharey=True, sharex=True)
axes = axes.flatten()
idx = np.arange(0,n_updates,n_updates//5)

for i in range(len(axes)):
    sns.distplot(reps[idx[i]], hist=False, ax=axes[i])
    axes[i].set_xlabel('z')
    axes[i].set_title(str(idx[i])+' updates')
axes[0].set_ylabel('p(z)')

plt.tight_layout()
plt.show()
