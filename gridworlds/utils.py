import numpy as np

def pos2xy(pos):
    return np.asarray(pos)[::-1]

def manhattan_dist(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return np.abs(x2-x1) + np.abs(y2-y1)
