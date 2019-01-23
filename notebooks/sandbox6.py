# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import scipy.stats
import scipy.ndimage.filters
import torch
from tqdm import tqdm

from notebooks.featurenet import FeatureNet
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld

#% Generate starting states
# env = GridWorld(rows=3,cols=3)
env =   TestWorld()
# env.add_random_walls(10)
# env.plot()
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
c0 = s0[:,0]*env._cols+s0[:,1]
s1 = np.asarray(states[1:,:])
a = np.asarray(actions)

np.sum(np.all(s0==s1,axis=1))

sigma = 0.1
x0 = s0 + sigma * np.random.randn(n_samples,2)
x1 = x0 + np.asarray(s1 - s0) + sigma/2 * np.random.randn(n_samples,2)
# x1 = s1 + sigma * np.random.randn(n_samples,2)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(231)
ax.scatter(x0[:,0],x0[:,1],c=c0, cmap='Set3')
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([])
plt.yticks([])
ax.set_title('states (t)')
# plt.show()

ax = fig.add_subplot(233)
ax.scatter(x1[:,0],x1[:,1],c=c0, cmap='Set3')
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([])
plt.yticks([])
ax.set_title('states (t+1)')

#% Entangle variables
def entangle(x):
    bins = 3*env._rows
    digitized = scipy.stats.binned_statistic_2d(x[:,0],x[:,1],np.arange(n_samples), bins=bins, expand_binnumbers=True)[-1].transpose()
    u = np.zeros([n_samples,bins,bins])
    for i in range(n_samples):
        u[i,digitized[i,0]-1,digitized[i,1]-1] = 1
    u = scipy.ndimage.filters.gaussian_filter(u, sigma=.6, truncate=1., mode='nearest')
    return u

u0 = entangle(x0)
u1 = entangle(x1)

ax = fig.add_subplot(232)
ax.imshow(u0[-1])
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
# plt.xlabel('u')
# plt.ylabel('v')
plt.xticks([])
plt.yticks([])
ax.set_title('observations (t)')

#%% Learn inv dynamics
input_shape = u0.shape[1:]
fnet = FeatureNet(n_actions=4, input_shape=input_shape, n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=0.001)
fnet.print_summary()

#%%

test_x0 = torch.as_tensor(u0[-n_samples//10:,:], dtype=torch.float32)
test_x1 = torch.as_tensor(u1[-n_samples//10:,:], dtype=torch.float32)
test_a  = torch.as_tensor(a[-n_samples//10:], dtype=torch.int)
test_c  = c0[-n_samples//10:]


with torch.no_grad():
    tx0 = torch.as_tensor(u0, dtype=torch.float32)
    tx1 = torch.as_tensor(u1, dtype=torch.float32)
    ta  = torch.as_tensor(a, dtype=torch.long)
    z0 = fnet.phi(tx0)
    z1 = fnet.phi(tx1)
    z1_hat = fnet.fwd_model(z0, ta)
    a_hat = fnet.inv_model(z0,z1)

    inv_loss = fnet.compute_inv_loss(a_logits=a_hat, a=ta)
    fwd_loss = fnet.compute_fwd_loss(z1, z1_hat)

ax = fig.add_subplot(234)
inv_x = z0.numpy()[:,0]
inv_y = z0.numpy()[:,1]
inv_sc = ax.scatter(inv_x,inv_y,c=c0, cmap='Set3')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.xlabel(r'$z_0$')
plt.ylabel(r'$z_1$')
plt.xticks([])
plt.yticks([])
ax.set_title(r'$\phi(x_t)$')

ax = fig.add_subplot(235)
fwd_x = z1_hat.numpy()[:,0]
fwd_y = z1_hat.numpy()[:,1]
fwd_sc = ax.scatter(fwd_x,fwd_y,c=c0, cmap='Set3')
tframe = ax.text(-0.25, .7, 'frame = '+str(0))
tinv = ax.text(-0.75, .5, 'inv_loss = '+str(inv_loss.numpy()))
tfwd = ax.text(-0.75, .3, 'fwd_loss = '+str(fwd_loss.numpy()))
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.xlabel(r'$z_0$')
plt.ylabel(r'$z_1$')
plt.xticks([])
plt.yticks([])
ax.set_title(r'$T(\phi(x_t),a_t)$')

ax = fig.add_subplot(236)
true_x = z1.numpy()[:,0]
true_y = z1.numpy()[:,1]
true_sc = ax.scatter(true_x,true_y,c=c0, cmap='Set3')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.xlabel(r'$z_0$')
plt.ylabel(r'$z_1$')
plt.xticks([])
plt.yticks([])
ax.set_title(r'$\phi(x_{t+1})$')


def get_batch(x0, x1, a, batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta

get_next_batch = lambda: get_batch(u0[:n_samples//2,:], u1[:n_samples//2,:], a[:n_samples//2], batch_size=batch_size)

batch_size = 1024
n_frames = 200
def animate(i, n_inv_steps=10, n_fwd_steps=1, plot_every=10):
    for _ in range(plot_every):
        for _ in range(n_inv_steps):
            tx0, tx1, ta = get_next_batch()
            fnet.train_inv_batch(tx0, tx1, ta)
        for _ in range(n_fwd_steps):
            tx0, tx1, ta = get_next_batch()
            fnet.train_fwd_batch(tx0, tx1, ta)

    with torch.no_grad():
        tx0 = torch.as_tensor(u0, dtype=torch.float32)
        tx1 = torch.as_tensor(u1, dtype=torch.float32)
        ta  = torch.as_tensor(a, dtype=torch.long)
        z0 = fnet.phi(tx0)
        z1 = fnet.phi(tx1)
        z1_hat = fnet.fwd_model(z0, ta)
        a_hat = fnet.inv_model(z0,z1)

        inv_loss = fnet.compute_inv_loss(a_logits=a_hat, a=ta)
        fwd_loss = fnet.compute_fwd_loss(z1, z1_hat)

        inv_sc.set_offsets(z0.numpy())
        fwd_sc.set_offsets(z1_hat.numpy())
        true_sc.set_offsets(z1.numpy())

        tframe.set_text('frame = '+str(i))
        tinv.set_text('inv_loss = '+str(inv_loss.numpy()))
        tfwd.set_text('fwd_loss = '+str(fwd_loss.numpy()))

#%%

# --- Watch live ---
plt.waitforbuttonpress()
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=n_frames, interval=1, repeat=False)
plt.show()

# --- Save video to file ---
# Writer = matplotlib.animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Cam Allen'), bitrate=256)
# ani.save('video.mp4', writer=writer)
