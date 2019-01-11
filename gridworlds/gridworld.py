import numpy as np
import matplotlib.pyplot as plt
from .grid import basicgrid, taxigrid, testgrid
from .utils import pos2xy
from .objects.agent import Agent

class GridWorld(basicgrid.BasicGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent()
        self.action_map = {
            0: basicgrid.LEFT,
            1: basicgrid.RIGHT,
            2: basicgrid.UP,
            3: basicgrid.DOWN
        }
        self.n_actions = len(self.action_map.keys())
        self.agent.position = np.asarray((0,0), dtype=int)

    def step(self, action):
        assert(action in range(self.n_actions))
        direction = self.action_map[action]
        if not self.has_wall(self.agent.position, direction):
            self.agent.position += direction

    def plot(self):
        ax = super().plot()
        self.agent.plot(ax)
        return ax

class TestWorld(testgrid.TestGrid, GridWorld):
    pass
