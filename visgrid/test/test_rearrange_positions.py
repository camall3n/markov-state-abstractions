#%%
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seeding
import sys
import time
import torch
from tqdm import tqdm

from markov_abstr.visgrid.models.featurenet import FeatureNet
from markov_abstr.visgrid.models.autoencoder import AutoEncoder
from markov_abstr.visgrid.repvis import RepVisualization, CleanVisualization
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld
from visgrid.utils import get_parser, MI
from visgrid.sensors import *

parser = get_parser()
if 'ipykernel' in sys.argv[0]:
    sys.argv = []
# yapf: disable
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
parser.add_argument('-n', '--n_updates', type=int, default=10000, help='Number of training updates')
parser.add_argument('-r', '--rows', type=int, default=7, help='Number of gridworld rows')
parser.add_argument('-c', '--cols', type=int, default=4, help='Number of gridworld columns')
parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
# yapf: enable
args = parser.parse_args()

seeding.seed(args.seed, np, torch)

#%% ------------------ Define MDP ------------------
env = GridWorld(rows=args.rows, cols=args.cols)
cmap = None

sensor = SensorChain([
    RearrangeXYPositionsSensor((env._rows, env._cols))
    # OffsetSensor(offset=(0.5, 0.5)),
    # NoisySensor(sigma=0.05),
    # ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
    # # ResampleSensor(scale=2.0),
    # BlurSensor(sigma=0.6, truncate=1.),
    # NoisySensor(sigma=0.01)
])
image_sensor = ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=1)

#%% ------------------ Generate experiences ------------------
n_samples = 50000
# states = [env.get_state()]
# actions = []
fig = plt.figure()
fig.show()

for t in range(n_samples):
    s = env.get_state()
    x = sensor.observe([s])[0]
    o = np.concatenate((image_sensor.observe(s), image_sensor.observe(x)), axis=1)
    plt.imshow(o)
    fig.canvas.draw()
    fig.canvas.flush_events()
    while True:
        # a = np.random.choice(env.actions)
        a = {
            'w': 2,
            's': 3,
            'a': 0,
            'd': 1,
        }[input('Move with WASD: ')]
        if env.can_run(a):
            break
    s, _, _ = env.step(a)
    # states.append(s)
    # actions.append(a)
# states = np.stack(states)
# s0 = np.asarray(states[:-1, :])
# c0 = s0[:, 0] * env._cols + s0[:, 1]
# s1 = np.asarray(states[1:, :])
# a = np.asarray(actions)

ds = list(map(np.linalg.norm, s1 - s0))

#%% ------------------ Define sensor ------------------
x0 = sensor.observe(s0)
x1 = sensor.observe(s1)
dx = list(map(np.linalg.norm, x1 - x0))

#%% ------------------ Plot ds vs dx ------------------
# import matplotlib.pyplot as plt
# plt.plot(x0[:, 0], x1[:, 1])
# plt.plot(s0[:, 0], s0[:, 1])
# plt.show()

#%%

for s, x in zip(s0[:20], x0):
    o = np.concatenate((image_sensor.observe(s), image_sensor.observe(x)), axis=1)
    plt.imshow(o)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)

# %%
