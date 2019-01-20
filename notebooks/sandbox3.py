# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import torch
from tqdm import tqdm

import notebooks.nnutils as nnutils
from gridworlds.domain.gridworld.gridworld import GridWorld

#%% Generate starting states
env = GridWorld(rows=3,cols=3)
n_samples = 20000
states = [env.get_state()]
a = np.random.choice(env.actions, size=n_samples, replace=True)
for t in range(n_samples):
    s, _, _ = env.step(a[t])
    states.append(s)
states = np.stack(states)
s0 = np.asarray(states[:-1,:])
c0 = s0[:,0]*3+s0[:,1]
s1 = np.asarray(states[1:,:])

sigma = 0.1
x0 = s0 + sigma * np.random.randn(n_samples,2) - (1,1)
x1 = x0 + np.asarray(s1 - s0) + sigma/2 * np.random.randn(n_samples,2)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(221)
ax.scatter(x0[:,0],x0[:,1],c=c0, cmap='tab10')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([])
plt.yticks([])
ax.set_title('states (t)')
# plt.show()

ax = fig.add_subplot(222)
ax.scatter(x1[:,0],x1[:,1],c=c0, cmap='tab10')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([])
plt.yticks([])
ax.set_title('states (t+1)')

#% Entangle variables
def entangle(x):
    u = np.zeros_like(x)
    u[:,0] = (x[:,0]+x[:,1])/2
    u[:,1] = (x[:,0]-x[:,1])/2
    return u
# def entangle(x):
#     return x

u0 = entangle(x0)
u1 = entangle(x1)

ax = fig.add_subplot(223)
ax.scatter(u0[:,0], u0[:,1],c=c0, cmap='tab10')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xlabel('u')
plt.ylabel('v')
ax.set_title('observations (t)')

# plt.scatter(u1[:,0], u1[:,1], c=s0)
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# plt.xlabel('u')
# plt.ylabel('v')
# plt.title('observations (t+1)')
# plt.show()

#%% Learn inv dynamics
fnet = nnutils.FeatureNet(n_actions=4, n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=0.002)
# fnet.print_summary()

ax = fig.add_subplot(224)
x, y = [], []
with torch.no_grad():
    tx0 = torch.as_tensor(u0, dtype=torch.float32)
    tx1 = torch.as_tensor(u1, dtype=torch.float32)
    a_hat = fnet.predict_a(tx0,tx1).numpy()
    accuracy = np.sum(a_hat == a)/len(a)
    z0 = fnet.phi(tx0).numpy()
x = z0[:,0]
y = z0[:,1]
sc = ax.scatter(x,y,c=c0, cmap='tab10')
t = ax.text(-0.5, .5, 'accuracy = '+str(accuracy))
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.xlabel(r'$z_0$')
plt.ylabel(r'$z_1$')
plt.xticks([])
plt.yticks([])
ax.set_title('representation (t)')

def get_batch(x0, x1, a, batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta

batch_size = 1024

def animate(i, steps_per_frame=5):
    for i in range(steps_per_frame):
        tx0, tx1, ta = get_batch(u0[:n_samples//2,:], u1[:n_samples//2,:], a[:n_samples//2], batch_size=batch_size)
        loss = fnet.train_batch(tx0, tx1, ta)

    with torch.no_grad():
        tx0 = torch.as_tensor(u0, dtype=torch.float32)
        tx1 = torch.as_tensor(u1, dtype=torch.float32)
        ta  = torch.as_tensor(a, dtype=torch.int)
        a_hat = fnet.predict_a(tx0,tx1).numpy()
        accuracy = np.sum(a_hat == a)/len(a)
        z0 = fnet.phi(tx0).numpy()
        z1 = fnet.phi(tx1).numpy()
        sc.set_offsets(z0)
        t.set_text('accuracy = '+str(accuracy))

# --- Watch live ---
plt.waitforbuttonpress()
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=50, interval=33, repeat=False)
plt.show()

# --- Save video to file ---
# ani = matplotlib.animation.FuncAnimation(fig, lambda i: animate(i, steps_per_frame=20), frames=30, interval=1, repeat=False)
# Writer = matplotlib.animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Cam Allen'), bitrate=1024)
# ani.save('representation3.mp4', writer=writer)
