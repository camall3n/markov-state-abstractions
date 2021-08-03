import random
import os

import matplotlib.pyplot as plt
import numpy as np
import seeding
from tqdm import tqdm

from visgrid.gridworld import MazeWorld

rows = 6
cols = 6

for seed in tqdm(range(1, 301)):
    seeding.seed(seed, np, random)
    env = MazeWorld(rows=rows, cols=cols)
    maze_dir = 'gridworlds/domain/gridworld/mazes/mazes_{}x{}/seed-{:03d}/'.format(
        rows, cols, seed)
    os.makedirs(maze_dir, exist_ok=True)

    txt_file = 'maze-{}.txt'.format(seed)
    env.save(maze_dir + txt_file)

    png_file = 'maze-{}.png'.format(seed)
    env.plot()
    plt.savefig(maze_dir + png_file, facecolor='white', edgecolor='none')
    plt.close()
