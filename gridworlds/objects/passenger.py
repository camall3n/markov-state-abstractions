import numpy as np
import matplotlib.pyplot as plt
from .basicsprite import BasicSprite
from ..utils import pos2xy

class Passenger(BasicSprite):
    def __init__(self, position=(0,0), name='Alice'):
        self.position = np.asarray(position)
        self.name = name
        self.intaxi = False

    def plot(self, ax):
        x,y = pos2xy(self.position)+(0.5, 0.5)
        if self.intaxi:
            c = plt.Circle((x,y), 0.2, color='k', fill=True, linewidth=1)
            ax.add_patch(c)
            plt.text(x, y, self.name[0], fontsize=12, color='white',
                horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(x, y, self.name[0], fontsize=14,
                horizontalalignment='center', verticalalignment='center')
