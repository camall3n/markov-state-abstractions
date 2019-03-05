import numpy as np
import random
import torch

def manhattan_dist(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return np.abs(x2-x1) + np.abs(y2-y1)

def reset_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
