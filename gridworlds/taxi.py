import numpy as np
import matplotlib.pyplot as plt
from .grid import taxigrid
from .gridworld import GridWorld
from .objects.passenger import Passenger
from .objects.depot import Depot

class TaxiWorld(taxigrid.TaxiGrid, GridWorld):
    def __init__(self, *args, **kwargs):
        self.passenger = Passenger()
        self.depot = Depot()

        super().__init__(*args, **kwargs)
        self.n_actions = len(self.action_map.keys())+2
        self.reset()

    def reset(self):
        self.agent.position = (2,2)
        self.passenger.position = (0,0)
        self.depot.position = (0,0)

    def step(self, action):
        if action < 4:
            super().step(action)
        elif action == 4:
            pass
        else:
            pass

    def plot(self):
        ax = super().plot()
        self.passenger.plot(ax)
        self.depot.plot(ax)
