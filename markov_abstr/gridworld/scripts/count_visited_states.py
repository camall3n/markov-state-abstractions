import imageio
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
import torch
from tqdm import tqdm

from ..models.featurenet import FeatureNet
from ..models.autoencoder import AutoEncoder
from ..repvis import RepVisualization, CleanVisualization
from ..visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from ..visgrid.utils import reset_seeds, get_parser, MI
from ..visgrid.sensors import *
from ..visgrid.gridworld.distance_oracle import DistanceOracle

class Args:
    pass

args = Args()

args.rows = 6
args.cols = 6
args.walls = 'maze'

data = []
for seed in tqdm(range(1, 301)):

    reset_seeds(seed)
    if args.walls == 'maze':
        env = MazeWorld.load_maze(rows=args.rows, cols=args.cols, seed=seed)
    elif args.walls == 'spiral':
        env = SpiralWorld(rows=args.rows, cols=args.cols)
    elif args.walls == 'loop':
        env = LoopWorld(rows=args.rows, cols=args.cols)
    else:
        env = GridWorld(rows=args.rows, cols=args.cols)

    #% ------------------ Generate experiences ------------------
    n_samples = 20000
    states = [env.get_state()]
    actions = []
    for t in range(n_samples):
        a = np.random.choice(env.actions)
        s, _, _ = env.step(a)
        states.append(s)
        actions.append(a)
    states = np.stack(states)
    s0 = np.asarray(states[:-1, :])
    c0 = s0[:, 0] * env._cols + s0[:, 1]
    s1 = np.asarray(states[1:, :])
    a = np.asarray(actions)

    _, state_counts = np.unique(states, axis=0, return_counts=True)
    seed_data = pd.DataFrame(state_counts, columns=['counts'])
    seed_data['seed'] = seed
    data.append(seed_data)

data = pd.concat(data)
sns.boxplot(data=data, x='seed', y='counts')
plt.ylim([0, 1200])
plt.show()
plt.figure()
plt.hist(data['counts'], bins=100)
plt.show()
