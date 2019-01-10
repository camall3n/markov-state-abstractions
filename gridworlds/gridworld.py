import numpy as np
import matplotlib.pyplot as plt
from .grid import basicgrid, taxigrid, testgrid

class GridWorld(basicgrid.BasicGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_map = {
            0: basicgrid.LEFT,
            1: basicgrid.RIGHT,
            2: basicgrid.UP,
            3: basicgrid.DOWN
        }
        self.n_actions = len(self.action_map.keys())
        self.reset()

    def reset(self):
        self.agent_pos = np.asarray((0,0), dtype=int)

    def step(self, action):
        assert(action in range(self.n_actions))
        direction = self.action_map[action]
        if not self.has_wall(self.agent_pos, direction):
            self.agent_pos += direction

    def plot(self):
        ax = super().plot()
        xy = self.pos2xy(self.agent_pos)
        c = plt.Circle(xy, 0.2, color='k', fill=False, linewidth=1)
        ax.add_patch(c)

    def pos2xy(self, pos):
        pos = np.asarray(pos)
        return (pos*(-1,1)+(self._rows,0)+(-0.5,0.5))[::-1]

class TaxiWorld(taxigrid.TaxiGrid, GridWorld):
    pass

class TestWorld(testgrid.TestGrid, GridWorld):
    pass
