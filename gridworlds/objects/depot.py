import numpy as np
import matplotlib.pyplot as plt
from .basicsprite import BasicSprite
from ..utils import pos2xy

class Depot(BasicSprite):
    def __init__(self, position=(0,0), color='red'):
        self.position = np.asarray(position)
        self.color = color

    def plot(self, ax):
        xy = pos2xy(self.position)+(0.1,0.1)
        colorname = 'goldenrod' if self.color=='yellow' else self.color
        c = plt.Rectangle(xy, 0.8, 0.8, color=colorname, fill=False, linewidth=1)
        ax.add_patch(c)
