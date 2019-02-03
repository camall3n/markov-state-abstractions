import numpy as np
import matplotlib.pyplot as plt
from .basesprite import BaseSprite, pos2xy

class Passenger(BaseSprite):
    def __init__(self, position=(0,0), name='Alice'):
        self.position = np.asarray(position)
        self.name = name
        self.intaxi = False

    def plot(self, ax):
        x,y = pos2xy(self.position)+(0.5, 0.5)
        if self.intaxi:
            c = plt.Circle((x,y), 0.2, color='k', fill=True, linewidth=1)
            ax.add_patch(c)
            textcolor = 'white'
        else:
            textcolor = 'black'
        ax.text(x, y, self.name[0], fontsize=12, color=textcolor,
            horizontalalignment='center', verticalalignment='center')
