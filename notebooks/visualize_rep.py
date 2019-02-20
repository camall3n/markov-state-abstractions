import imageio
import numpy as np
import random
import scipy.stats
import scipy.ndimage.filters
import torch
from tqdm import tqdm

from notebooks.featurenet import FeatureNet
from notebooks.repvis import RepVisualization
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld

#%% ------------------ Define MDP ------------------
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

env = GridWorld(rows=7,cols=4)
# env = RingWorld(2,4)
# env = TestWorld()
# env.add_random_walls(10)
# env.plot()

# cmap = 'Set3'
cmap = None

#%% ------------------ Generate experiences ------------------
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

#%% ------------------ Define sensors ------------------
sigma = 0.1
x0 = s0 + sigma * np.random.randn(n_samples,2)
x1 = x0 + np.asarray(s1 - s0) + sigma/2 * np.random.randn(n_samples,2)
# x1 = s1 + sigma * np.random.randn(n_samples,2)

def entangle(x):
    bins = (3*env._rows, 3*env._cols)
    digitized = scipy.stats.binned_statistic_2d(x[:,0],x[:,1],np.arange(n_samples), bins=bins, expand_binnumbers=True)[-1].transpose()
    u = np.zeros([n_samples,bins[0],bins[1]])
    for i in range(n_samples):
        u[i,digitized[i,0]-1,digitized[i,1]-1] = 1
    u = scipy.ndimage.filters.gaussian_filter(u, sigma=.6, truncate=1., mode='nearest')
    return u

u0 = entangle(x0)
u1 = entangle(x1)

#%% ------------------ Setup experiment ------------------
n_steps = 1000
n_frames = 100
n_updates_per_frame = n_steps // n_frames

batch_size = 1024
n_inv_steps_per_update = 10
n_fwd_steps_per_update = 1
n_disentangle_steps_per_update = 1
n_entropy_steps_per_update = 0

fnet = FeatureNet(n_actions=4, input_shape=u0.shape[1:], n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=0.001)
fnet.print_summary()

n_test_samples = 2000
test_x0 = x0[-n_test_samples:,:]
test_x1 = x1[-n_test_samples:,:]
test_u0 = torch.as_tensor(u0[-n_test_samples:,:], dtype=torch.float32)
test_u1 = torch.as_tensor(u1[-n_test_samples:,:], dtype=torch.float32)
test_a  = torch.as_tensor(a[-n_test_samples:], dtype=torch.long)
test_c  = c0[-n_test_samples:]
obs = test_u0[-1]

repvis = RepVisualization(env, test_x0, test_x1, obs, colors=test_c, cmap=cmap)

def get_batch(x0, x1, a, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta

get_next_batch = lambda: get_batch(u0[:n_samples//2,:], u1[:n_samples//2,:], a[:n_samples//2])

def test_rep(fnet):
    with torch.no_grad():
        fnet.eval()
        z0 = fnet.phi(test_u0)
        z1 = fnet.phi(test_u1)
        z1_hat = fnet.fwd_model(z0, test_a)
        a_hat = fnet.inv_model(z0, z1)

        inv_loss = fnet.compute_inv_loss(a_logits=a_hat, a=test_a)
        fwd_loss = fnet.compute_fwd_loss(z0, z1, z1_hat)
        dis_loss = fnet.compute_disentanglement_loss(z0, z1)
        ent_loss = fnet.compute_entropy_loss(z0, z1, test_a)
    results = [z0, z1_hat, z1, inv_loss, fwd_loss, dis_loss, ent_loss, test_a, a_hat]
    return [r.numpy() for r in results]

#%% ------------------ Run Experiment ------------------
data = []
for frame_idx in tqdm(range(n_frames+1)):
    for _ in range(n_updates_per_frame):
        for _ in range(n_inv_steps_per_update):
            tx0, tx1, ta = get_next_batch()
            fnet.train_batch(tx0, tx1, ta, model='inv')
        for _ in range(n_fwd_steps_per_update):
            tx0, tx1, ta = get_next_batch()
            fnet.train_batch(tx0, tx1, ta, model='fwd')
        for _ in range(n_disentangle_steps_per_update):
            tx0, tx1, ta = get_next_batch()
            fnet.train_batch(tx0, tx1, ta, model='disentanglement')
        for _ in range(n_entropy_steps_per_update):
            tx0, tx1, ta = get_next_batch()
            fnet.train_batch(tx0, tx1, ta, model='entropy')

    frame = repvis.update_plots(frame_idx*n_updates_per_frame, *test_rep(fnet))
    data.append(frame)

imageio.mimwrite('video.mp4', data, fps=15)
