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
from gridworlds.domain.gridworld.gridworld import GridWorld

#% Generate starting states
env = GridWorld(rows=3,cols=3)

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

sigma = 0.1
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

#% Entangle variables
def entangle(x):
    bins = 2*env._rows
    digitized = scipy.stats.binned_statistic_2d(x[:,0],x[:,1],np.arange(n_samples), bins=bins, expand_binnumbers=True)[-1].transpose()
    u = np.zeros([n_samples,bins,bins])
    for i in range(n_samples):
        u[i,digitized[i,0]-1,digitized[i,1]-1] = 1
    u = scipy.ndimage.filters.gaussian_filter(u, sigma=.6, truncate=1., mode='nearest')
    return u

u0 = entangle(x0)
u1 = entangle(x1)

ax = fig.add_subplot(223)
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
fnet = nnutils.FeatureNet(n_actions=4, input_shape=input_shape, n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=0.001)
fnet.print_summary()

#%%

test_x0 = torch.as_tensor(u0[-n_samples//10:,:], dtype=torch.float32)
test_x1 = torch.as_tensor(u1[-n_samples//10:,:], dtype=torch.float32)
test_a  = torch.as_tensor(a[-n_samples//10:], dtype=torch.int)
test_c  = c0[-n_samples//10:]

ax = fig.add_subplot(224)
x, y = [], []
with torch.no_grad():
    a_hat = fnet.predict_a(test_x0,test_x1).numpy()
    accuracy = np.sum(a_hat == test_a.numpy())/len(test_a.numpy())
    z0 = fnet.phi(test_x0).numpy()
x = z0[:,0]
y = z0[:,1]
sc = ax.scatter(x,y,c=test_c)
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
def animate(i, steps_per_frame=20):
    loss = 0
    for i in range(steps_per_frame):
        tx0, tx1, ta = get_batch(u0[:n_samples//2,:], u1[:n_samples//2,:], a[:n_samples//2], batch_size=batch_size)
        loss += fnet.train_batch(tx0, tx1, ta).detach().numpy()

    with torch.no_grad():

        a_hat = fnet.predict_a(test_x0,test_x1).numpy()
        accuracy = np.sum(a_hat == test_a.numpy())/len(test_a.numpy())
        z0 = fnet.phi(test_x0).numpy()
        z1 = fnet.phi(test_x1).numpy()
        sc.set_offsets(z0)
        t.set_text('accuracy = '+str(accuracy))

#%%

# --- Watch live ---
plt.waitforbuttonpress()
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=50, interval=33, repeat=False)
plt.show()

# --- Save video to file ---
# ani = matplotlib.animation.FuncAnimation(fig, lambda i: animate(i, steps_per_frame=20), frames=50, interval=1, repeat=False)
# Writer = matplotlib.animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Cam Allen'), bitrate=1024)
# ani.save('representation5.mp4', writer=writer)
