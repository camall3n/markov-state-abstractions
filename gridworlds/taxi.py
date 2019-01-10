import numpy as np
import matplotlib.pyplot as plt
from .grid import taxigrid
from .gridworld import GridWorld
from .objects.passenger import Passenger
from .objects.depot import Depot

class TaxiWorld(taxigrid.TaxiGrid, GridWorld):
    def __init__(self, *args, **kwargs):
        self.passenger = Passenger()
        self.depots = dict([(color, Depot(color=color)) for color in ['red', 'blue', 'green', 'yellow']])
        self.depots['red'].position = (0,0)
        self.depots['yellow'].position = (4,0)
        self.depots['blue'].position = (4,3)
        self.depots['green'].position = (0,4)

        super().__init__(*args, **kwargs)
        self.n_actions = len(self.action_map.keys())+2
        self.reset()

    def reset(self):
        self.agent.position = (2,2)
        self.passenger.position = (0,0)

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
        for _, depot in self.depots.items():
            depot.plot(ax)
