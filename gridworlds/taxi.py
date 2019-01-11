import numpy as np
import matplotlib.pyplot as plt
import random
from .grid.taxigrid import TaxiGrid
from .gridworld import GridWorld
from .objects.passenger import Passenger
from .objects.depot import Depot

class TaxiWorld(TaxiGrid, GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passenger = Passenger(name='Passenger')
        self.depots = dict([(color, Depot(color=color)) for color in ['red', 'blue', 'green', 'yellow']])
        self.depots['red'].position = (0,0)
        self.depots['yellow'].position = (4,0)
        self.depots['blue'].position = (4,3)
        self.depots['green'].position = (0,4)

        self.n_actions = len(self.action_map.keys())+1
        depotnames = list(self.depots.keys())
        random.shuffle(depotnames)
        self.agent.position = self.depots[depotnames.pop()].position
        self.passenger.position = self.depots[depotnames.pop()].position

    def plot(self):
        ax = super().plot()
        for _, depot in self.depots.items():
            depot.plot(ax)
        self.passenger.plot(ax)

    def step(self, action):
        if action < 4:
            super().step(action)
            if self.passenger.intaxi:
                self.passenger.position = self.agent.position
        elif action == 4:# Interact
            if (self.agent.position == self.passenger.position).all():
                self.passenger.intaxi = not self.passenger.intaxi

class TaxiOfHanoi(TaxiWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passenger = None
        self.passengers = [Passenger(name=name) for name in ['Alice', 'Bob', 'Carol']]
        self.reset(seed)

    def reset(self):
        depotnames = list(self.depots.keys())
        random.shuffle(depotnames)
        for p in self.passengers:
            p.position = self.depots[depotnames.pop()].position
        self.agent.position = self.depots[depotnames.pop()].position

    def plot(self):
        ax = super(TaxiWorld, self).plot()
        for _, depot in self.depots.items():
            depot.plot(ax)
        for p in (p for p in self.passengers if not p.intaxi):
            p.plot(ax)
        for p in (p for p in self.passengers if p.intaxi):
            p.plot(ax)

    def step(self, action):
        if action < 4:
            super(TaxiWorld, self).step(action)
            for p in self.passengers:
                if p.intaxi:
                    p.position = self.agent.position
                    break # max one passenger per taxi
        elif action == 4:# Interact
            if self.passenger is None:
                for p in self.passengers:
                    if (self.agent.position == p.position).all():
                        p.intaxi = True
                        self.passenger = p
                        break # max one passenger per taxi
            else:
                dropoff_clear = True
                for p in (p for p in self.passengers if p is not self.passenger):
                    if (p.position == self.passenger.position).all():
                        dropoff_clear = False
                        break
                if dropoff_clear:
                    self.passenger.intaxi = False
                    self.passenger = None
