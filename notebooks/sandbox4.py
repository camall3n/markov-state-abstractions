# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import torch
from tqdm import tqdm

from notebooks.featurenet import FeatureNet
from gridworlds.domain.gridworld.gridworld import GridWorld

#% Generate starting states
env = GridWorld(rows=6,cols=6)
env._grid[4,5] = 1
env._grid[4,7] = 1
env._grid[5,4] = 1
env._grid[5,8] = 1
env._grid[7,4] = 1
env._grid[7,8] = 1
env._grid[8,5] = 1
env._grid[8,7] = 1

#%%
n_samples = 20000
states = [env.get_state()]
actions = []
for t in range(n_samples):
    while True:
        a = np.random.choice(env.actions)
        if env.can_run(a):
            break
    s, _, _ = env.step(a)
    states.append(s)
    actions.append(a)
states = np.stack(states)
s0 = np.asarray(states[:-1,:])
c0 = s0[:,0]*env._rows+s0[:,1]
s1 = np.asarray(states[1:,:])
a = np.asarray(actions)

np.sum(np.all(s0==s1,axis=1))

sigma = 0.03
x0 = s0 + sigma * np.random.randn(n_samples,2)
x1 = x0 + np.asarray(s1 - s0) + sigma/2 * np.random.randn(n_samples,2)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(221)
ax.scatter(x0[:,0],x0[:,1],c=c0)
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([])
plt.yticks([])
ax.set_title('states (t)')
# plt.show()

ax = fig.add_subplot(222)
ax.scatter(x1[:,0],x1[:,1],c=c0)
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([])
plt.yticks([])
ax.set_title('states (t+1)')

#%% Entangle variables
def entangle(x):
    u = np.zeros_like(x)
    u[:,0] = (x[:,0]+x[:,1])
    u[:,1] = (x[:,0]-x[:,1])
    u -= np.mean(u, axis=0)
    u /= np.max(np.abs(u),axis=0)
    return u
# def entangle(x):
#     return x

u0 = entangle(x0)
u1 = entangle(x1)

ax = fig.add_subplot(223)
ax.scatter(u0[:,0], u0[:,1],c=c0)
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
plt.xlabel('u')
plt.ylabel('v')
# plt.xticks([])
# plt.yticks([])
ax.set_title('observations (t)')

# plt.scatter(u1[:,0], u1[:,1], c=s0)
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# plt.xlabel('u')
# plt.ylabel('v')
# plt.title('observations (t+1)')
# plt.show()

#%% Learn inv dynamics
fnet = FeatureNet(n_actions=4, n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=0.001, inv_steps_per_fwd=10)
fnet.print_summary()

#%%

test_x0 = torch.as_tensor(u0[-n_samples//10:,:], dtype=torch.float32)
test_x1 = torch.as_tensor(u1[-n_samples//10:,:], dtype=torch.float32)
test_a  = torch.as_tensor(a[-n_samples//10:], dtype=torch.int)
test_c  = c0[-n_samples//10:]

ax = fig.add_subplot(224)
x, y = [], []
with torch.no_grad():
    tx0 = torch.as_tensor(u0, dtype=torch.float32)
    tx1 = torch.as_tensor(u1, dtype=torch.float32)
    ta  = torch.as_tensor(a, dtype=torch.long)
    z0 = fnet.phi(tx0)
    z1 = fnet.phi(tx1)
    z1_hat = fnet.fwd_model(z0, ta)

    inv_a_hat = fnet.predict_a(z0,z1).numpy()
    inv_accuracy = np.sum(inv_a_hat == a)/len(a)
    fwd_a_hat = fnet.predict_a(z0,z1_hat).numpy()
    fwd_accuracy = np.sum(fwd_a_hat == a)/len(a)
x = z0.numpy()[:,0]
y = z0.numpy()[:,1]
sc = ax.scatter(x,y,c=c0)
tframe = ax.text(-0.25, .7, 'frame = '+str(0))
tinv = ax.text(-0.5, .5, 'inv_accuracy = '+str(inv_accuracy))
tfwd = ax.text(-0.5, .3, 'fwd_accuracy = '+str(fwd_accuracy))
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
n_frames = 200
def animate(i, steps_per_frame=1):
    for _ in range(steps_per_frame):
        tx0, tx1, ta = get_batch(u0[:n_samples//2,:], u1[:n_samples//2,:], a[:n_samples//2], batch_size=batch_size)
        loss = fnet.train_batch(tx0, tx1, ta)

    with torch.no_grad():
        tx0 = torch.as_tensor(u0, dtype=torch.float32)
        tx1 = torch.as_tensor(u1, dtype=torch.float32)
        ta  = torch.as_tensor(a, dtype=torch.long)
        z0 = fnet.phi(tx0)
        z1 = fnet.phi(tx1)
        z1_hat = fnet.fwd_model(z0, ta)

        inv_a_hat = fnet.predict_a(z0,z1).numpy()
        inv_accuracy = np.sum(inv_a_hat == a)/len(a)
        fwd_a_hat = fnet.predict_a(z0,z1_hat).numpy()
        fwd_accuracy = np.sum(fwd_a_hat == a)/len(a)

        sc.set_offsets(z0.numpy())

        tframe.set_text('frame = '+str(i))
        tinv.set_text('inv_accuracy = '+str(inv_accuracy))
        tfwd.set_text('fwd_accuracy = '+str(fwd_accuracy))

#%%

# --- Watch live ---
# plt.waitforbuttonpress()
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=n_frames, interval=1, repeat=False)
# plt.show()

# --- Save video to file ---
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Cam Allen'), bitrate=256)
ani.save('representation4.mp4', writer=writer)
