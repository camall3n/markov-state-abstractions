import numpy as np
import matplotlib.pyplot as plt
from .basesprite import BaseSprite, pos2xy

class Depot(BaseSprite):
    def __init__(self, position=(0,0), color='red'):
        self.position = np.asarray(position)
        self.color = color

    def plot(self, ax):
        xy = pos2xy(self.position)+(0.1,0.1)
        colorname = self.color
        colorname = 'gold' if colorname=='yellow' else colorname
        colorname = 'c' if colorname=='cyan' else colorname
        colorname = 'm' if colorname=='magenta' else colorname
        colorname = 'silver' if colorname in ['gray','grey'] else colorname
        c = plt.Rectangle(xy, 0.8, 0.8, color=colorname, fill=False, linewidth=1)
        ax.add_patch(c)
