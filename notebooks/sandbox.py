%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from tqdm import tqdm

from notebooks.featurenet import FeatureNet

#%% Generate starting states
sigma = 0.1
n_samples = 1000
x0 = sigma * np.random.randn(n_samples,2)
plt.scatter(x0[:,0],x0[:,1])
plt.xlim(-.5,1.5)
plt.ylim(-.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('s0')
plt.show()

#%% Generate actions and next states
actions = ['up','right']
directions = [(0,1), (1,0)]

a = np.random.choice(2, size=n_samples)
slip = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
direction = a ^ slip

dx = np.asarray([directions[d] for d in direction])
n = sigma * np.random.randn(n_samples,2)
x1 = x0 + dx + n

plt.scatter(x1[:,0],x1[:,1], c=a)
plt.xlim(-.5,1.5)
plt.ylim(-.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('s\'')
plt.show()

#%% Entangle variables
def entangle(x):
    u = np.zeros_like(x)
    u[:,0] = x[:,0]*x[:,1]+sigma*x[:,0]
    u[:,1] = np.cos(15*(x[:,1]*x[:,0]+sigma*x[:,1]))
    return u

u0 = entangle(x0)
u1 = entangle(x1)

plt.scatter(u0[:,0], u0[:,1],c=a)
plt.xlabel('u')
plt.ylabel('v')
plt.title('sensor(s)')
plt.show()

plt.scatter(u1[:,0], u1[:,1],c=a)
plt.xlabel('u')
plt.ylabel('v')
plt.title('sensor(s\')')
plt.show()

#%% Learn inv dynamics
fnet = FeatureNet(n_actions=2, n_latent_dims=2, lr=0.001)

def get_batch(x0, x1, a, batch_size=32):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta
def train(fnet):
    batch_size = 32
    running_loss = 0.0
    for i in tqdm(range(500)):
        loss = fnet.train_batch(*get_batch(x0, x1, a))
        running_loss += loss
        if i % 20 == 19:    # print every 20 mini-batches
            tqdm.write('[%d] loss: %.3f' %
                  (i + 1, running_loss / 20))
            running_loss = 0.0
train(fnet)

#%% Explain effects
with torch.no_grad():
    tx0 = torch.as_tensor(x0, dtype=torch.float32)
    tx1 = torch.as_tensor(x1, dtype=torch.float32)
    a_hat = fnet.predict_a(tx0,tx1).numpy()
    z0 = fnet.phi(tx0).numpy()
    z1 = fnet.phi(tx1).numpy()

plt.scatter(z0[:,0], z0[:,1],c=a_hat)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('sensor(s)')
plt.show()

plt.scatter(z1[:,0], z1[:,1],c=a_hat)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('sensor(s\')')
plt.show()
