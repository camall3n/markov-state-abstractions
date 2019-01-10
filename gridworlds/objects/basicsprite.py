import numpy as np
import matplotlib.pyplot as plt
from ..utils import pos2xy

class BasicSprite:
    def __init__(self, position=(0,0)):
        self.position = np.asarray(position)

    def __setattr__(self, name, value):
        if name=='position':
            value = np.asarray(value, dtype=int)
        super().__setattr__(name, value)

    def plot(self, ax):
        xy = pos2xy(self.position)+(0.5,0.5)
        c = plt.Circle(xy, 0.2, color='k', fill=False, linewidth=1)
        ax.add_patch(c)
