import numpy as np
import matplotlib.pyplot as plt

from . import grid
from .objects.agent import Agent
from .objects.depot import Depot

class GridWorld(grid.BaseGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent()
        self.actions = [i for i in range(4)]
        self.action_map = {0: grid.LEFT, 1: grid.RIGHT, 2: grid.UP, 3: grid.DOWN}
        self.agent.position = np.asarray((0, 0), dtype=int)
        self.goal = None

    def reset_agent(self):
        self.agent.position = self.get_random_position()
        at = lambda x, y: np.all(x.position == y.position)
        while (self.goal is not None) and at(self.agent, self.goal):
            self.agent.position = self.get_random_position()

    def reset_goal(self):
        if self.goal is None:
            self.goal = Depot()
        self.goal.position = self.get_random_position()
        self.reset_agent()

    def step(self, action):
        assert (action in range(4))
        direction = self.action_map[action]
        if not self.has_wall(self.agent.position, direction):
            self.agent.position += direction
        s = self.get_state()
        if self.goal:
            at_goal = np.all(self.agent.position == self.goal.position)
            r = 0 if at_goal else -1
            done = True if at_goal else False
        else:
            r = 0
            done = False
        return s, r, done

    def can_run(self, action):
        assert (action in range(4))
        direction = self.action_map[action]
        return False if self.has_wall(self.agent.position, direction) else True

    def get_state(self):
        return np.copy(self.agent.position)

    def plot(self, ax=None):
        ax = super().plot(ax)
        if self.agent:
            self.agent.plot(ax)
        if self.goal:
            self.goal.plot(ax)
        return ax
