import imageio
import numpy as np
import random
import torch
from tqdm import tqdm

from notebooks.featurenet import FeatureNet
from notebooks.repvis import RepVisualization
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld
from gridworlds.utils import reset_seeds
from notebooks.sensor import *

#%% ------------------ Define MDP ------------------
seed = 3
reset_seeds(seed)

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

#%% ------------------ Define sensor ------------------
sensor = SensorChain([
            NoisySensor(sigma=0.05),
            ImageSensor(size=(3*env._rows, 3*env._cols)),
            BlurSensor(sigma=0.6, truncate=1.),
        ])

x0 = sensor.observe(s0)
x1 = sensor.observe(s1)

#%% ------------------ Setup experiment ------------------
n_steps = 300
n_frames = 30
n_updates_per_frame = n_steps // n_frames

batch_size = 1024
n_inv_steps_per_update = 10
n_fwd_steps_per_update = 10
n_disentangle_steps_per_update = 0
n_entropy_steps_per_update = 0

fnet = FeatureNet(n_actions=4, input_shape=x0.shape[1:], n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=0.003)
fnet.print_summary()

n_test_samples = 2000
test_x0 = torch.as_tensor(x0[-n_test_samples:,:], dtype=torch.float32)
test_x1 = torch.as_tensor(x1[-n_test_samples:,:], dtype=torch.float32)
test_a  = torch.as_tensor(a[-n_test_samples:], dtype=torch.long)
test_c  = c0[-n_test_samples:]
obs = test_x0[-1]

repvis = RepVisualization(env, obs, batch_size=n_test_samples, n_dims=2, colors=test_c, cmap=cmap)

def get_batch(x0, x1, a, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta

get_next_batch = lambda: get_batch(x0[:n_samples//2,:], x1[:n_samples//2,:], a[:n_samples//2])

def test_rep(fnet, step):
    with torch.no_grad():
        fnet.eval()
        z0 = fnet.phi(test_x0)
        z1 = fnet.phi(test_x1)
        z1_hat = fnet.fwd_model(z0, test_a)
        a_hat = fnet.inv_model(z0, z1)

        inv_loss = fnet.compute_inv_loss(a_logits=a_hat, a=test_a)
        fwd_loss = fnet.compute_fwd_loss(z0, z1, z1_hat)
        dis_loss = fnet.compute_disentanglement_loss(z0, z1)
        ent_loss = fnet.compute_entropy_loss(z0, z1, test_a)

        text = '\n'.join([
            'updates = '+str(step),
            'inv_loss = '+str(inv_loss.numpy()),
            'fwd_loss = '+str(fwd_loss.numpy()),
            'dis_loss = '+str(dis_loss.numpy()),
            'ent_loss = '+str(ent_loss.numpy()),
        ])
    results = [z0, z1_hat, z1, test_a, a_hat]
    return [r.numpy() for r in results] + [text]

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

    frame = repvis.update_plots(*test_rep(fnet, frame_idx*n_updates_per_frame))
    data.append(frame)

imageio.mimwrite('video-{}.mp4'.format(seed), data, fps=15)
