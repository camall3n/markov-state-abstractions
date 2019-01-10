import numpy as np
import matplotlib.pyplot as plt
from .basicsprite import BasicSprite
from ..utils import pos2xy

class Agent(BasicSprite):
    def __init__(self, position=(0,0), name='Alice'):
        self.position = np.asarray(position)

    def plot(self, ax):
        xy = pos2xy(self.position)+(0.5,0.5)
        c = plt.Circle(xy, 0.3, color='k', fill=False, linewidth=1)
        ax.add_patch(c)
