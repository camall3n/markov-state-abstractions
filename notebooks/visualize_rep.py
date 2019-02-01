import imageio
import numpy as np
# import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.stats
import scipy.ndimage.filters
import torch
from tqdm import tqdm

from notebooks.featurenet import FeatureNet
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld

#%% ------------------ Define MDP ------------------
# env = GridWorld(rows=3,cols=3)
env =   TestWorld()
# env.add_random_walls(10)
# env.plot()

#%% ------------------ Generate experiences ------------------
n_samples = 20000
n_test_samples = 2000
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

#%% ------------------ Define sensors ------------------
sigma = 0.1
x0 = s0 + sigma * np.random.randn(n_samples,2)
x1 = x0 + np.asarray(s1 - s0) + sigma/2 * np.random.randn(n_samples,2)
# x1 = s1 + sigma * np.random.randn(n_samples,2)

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

#%% ------------------ Prepare visualization ------------------
fnet = FeatureNet(n_actions=4, input_shape=u0.shape[1:], n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=0.001)
fnet.print_summary()

fig = plt.figure(figsize=(10,6))
def plot_states(x, fig, subplot=111, colors=None, cmap=None, title=''):
    ax = fig.add_subplot(subplot)
    ax.scatter(x[:,0],x[:,1],c=colors, cmap=cmap)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks([])
    plt.yticks([])
    ax.set_title(title)

plot_states(x0, fig, subplot=231, colors=c0, cmap='Set3', title='states (t)')
plot_states(x1, fig, subplot=233, colors=c0, cmap='Set3', title='states (t+1)')

ax = fig.add_subplot(232)
ax.imshow(u0[-1])
plt.xticks([])
plt.yticks([])
ax.set_title('observations (t)')

test_x0 = torch.as_tensor(u0[-n_test_samples:,:], dtype=torch.float32)
test_x1 = torch.as_tensor(u1[-n_test_samples:,:], dtype=torch.float32)
test_a  = torch.as_tensor(a[-n_test_samples:], dtype=torch.long)
test_c  = c0[-n_test_samples:]

def test_rep(fnet, step):
    with torch.no_grad():
        fnet.eval()
        z0 = fnet.phi(test_x0)
        z1 = fnet.phi(test_x1)
        z1_hat = fnet.fwd_model(z0, test_a)
        a_hat = fnet.inv_model(z0, z1)

        inv_loss = fnet.compute_inv_loss(a_logits=a_hat, a=test_a)
        fwd_loss = fnet.compute_fwd_loss(z0, z1, z1_hat)
    return z0, z1_hat, z1, step, inv_loss, fwd_loss

z0, z1_hat, z1, step, inv_loss, fwd_loss = test_rep(fnet, step=0)

def plot_rep(z, fig, subplot=111, colors=None, cmap=None, title=''):
    ax = fig.add_subplot(subplot)
    z = z.numpy()
    x = z[:,0]
    y = z[:,1]
    sc = ax.scatter(x,y,c=test_c, cmap=cmap)
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.xlabel(r'$z_0$')
    plt.ylabel(r'$z_1$')
    plt.xticks([])
    plt.yticks([])
    ax.set_title(title)
    return ax, sc

_, inv_sc = plot_rep(z0, fig, subplot=234, colors=c0, cmap='Set3', title=r'$\phi(x_t)$')
ax, fwd_sc = plot_rep(z1_hat, fig, subplot=235, colors=c0, cmap='Set3', title=r'$T(\phi(x_t),a_t)$')
_, true_sc = plot_rep(z1, fig, subplot=236, colors=c0, cmap='Set3', title=r'$\phi(x_{t+1})$')

tstep = ax.text(-0.75, .7, 'updates = '+str(0))
tinv = ax.text(-0.75, .5, 'inv_loss = '+str(inv_loss.numpy()))
tfwd = ax.text(-0.75, .3, 'fwd_loss = '+str(fwd_loss.numpy()))

def update_plots(z0, z1_hat, z1, step, inv_loss, fwd_loss):
    inv_sc.set_offsets(z0.numpy())
    fwd_sc.set_offsets(z1_hat.numpy())
    true_sc.set_offsets(z1.numpy())

    tstep.set_text('updates = '+str(step))
    tinv.set_text('inv_loss = '+str(inv_loss.numpy()))
    tfwd.set_text('fwd_loss = '+str(fwd_loss.numpy()))

    fig.canvas.draw()
    fig.canvas.flush_events()

#%% ------------------ Run Experiment ------------------
n_steps = 2000
n_frames = n_steps // 20
n_updates_per_frame = n_steps // n_frames

batch_size = 1024
n_inv_steps_per_update = 10
n_fwd_steps_per_update = 1

def get_batch(x0, x1, a, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta

get_next_batch = lambda: get_batch(u0[:n_samples//2,:], u1[:n_samples//2,:], a[:n_samples//2])

fig.show()
pbar = tqdm(total=n_frames)
data = []
for frame in range(n_frames+1):
    for _ in range(n_updates_per_frame):
        for _ in range(n_inv_steps_per_update):
            tx0, tx1, ta = get_next_batch()
            fnet.train_batch(tx0, tx1, ta, model='inv')
        for _ in range(n_fwd_steps_per_update):
            tx0, tx1, ta = get_next_batch()
            fnet.train_batch(tx0, tx1, ta, model='fwd')

    update_plots(*test_rep(fnet, step=frame*n_updates_per_frame))

    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data.append(frame)
    pbar.update(1)

imageio.mimwrite('video.mp4', data, fps=15)
pbar.close()
