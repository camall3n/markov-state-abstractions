%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from tqdm import tqdm

import notebooks.nnutils as nnutils

#%% Generate starting states
sigma = 0.1
n_samples = 10000
states = ['a', 'b', 'c', 'd']
positions = {
    'a': (-1, 1),
    'b': (1, 1),
    'c': (1, -1),
    'd': (-1, -1),
}
s0 = np.random.choice(len(states),n_samples)
x0 = sigma * np.random.randn(n_samples,2) + np.asarray([positions[states[i]] for i in s0])

plt.scatter(x0[:,0],x0[:,1], c=s0)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('states (t)')
plt.show()

#%% Generate actions and next states
actions = ['cw','ccw']
a = np.random.choice(2, size=n_samples)
slip = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
direction = np.asarray([2*ac-1 for ac in a ^ slip])

s1 = (s0 + direction) % 4
x1 = x0 + np.asarray([positions[states[i1]] for i1 in s1]) - np.asarray([positions[states[i0]] for i0 in s0])

plt.scatter(x1[:,0],x1[:,1], c=s0)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('states (t+1)')
plt.show()

#%% Entangle variables
def entangle(x):
    u = np.zeros_like(x)
    u[:,0] = (x[:,0]+x[:,1])/2
    u[:,1] = (x[:,0]-x[:,1])/2
    return u
# def entangle(x):
#     return x

u0 = entangle(x0)
u1 = entangle(x1)

plt.scatter(u0[:,0], u0[:,1], c=s0)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('u')
plt.ylabel('v')
plt.title('observations (t)')
plt.show()

plt.scatter(u1[:,0], u1[:,1], c=s0)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('u')
plt.ylabel('v')
plt.title('observations (t+1)')
plt.show()

#%% Learn inv dynamics
fnet = nnutils.FeatureNet(n_actions=2, n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=128, lr=0.002)
# fnet.print_summary()

def get_batch(x0, x1, a, batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta

batch_size = 128
def train(fnet):
    running_loss = 0.0
    for i in tqdm(range(500)):
        tx0, tx1, ta = get_batch(u0[:n_samples//2,:], u1[:n_samples//2,:], a[:n_samples//2], batch_size=batch_size)
        loss = fnet.train_batch(tx0, tx1, ta)
        running_loss += loss
        if i % 50 == 49:
            plot_rep(ax)
            tqdm.write('[%d] loss: %.3f' %
                  (i + 1, running_loss / 20))
            running_loss = 0.0
train(fnet)

#%% Explain effects
with torch.no_grad():
    tx0 = torch.as_tensor(u0, dtype=torch.float32)
    tx1 = torch.as_tensor(u1, dtype=torch.float32)
    ta  = torch.as_tensor(a, dtype=torch.int)
    a_hat = fnet.predict_a(tx0,tx1).numpy()
    z0 = fnet.phi(tx0).numpy()
    z1 = fnet.phi(tx1).numpy()

plt.scatter(z0[:,0], z0[:,1], c=s0)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('sensor(s)')
plt.show()

plt.scatter(z1[:,0], z1[:,1], c=s0)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('sensor(s)')
plt.show()
