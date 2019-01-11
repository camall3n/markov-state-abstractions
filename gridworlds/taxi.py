import numpy as np
import matplotlib.pyplot as plt
import random
from .grid.basicgrid import BasicGrid
from .grid.taxigrid import TaxiGrid
from .gridworld import GridWorld
from .objects.passenger import Passenger
from .objects.depot import Depot

class TaxiWorld(TaxiGrid, GridWorld):
    def __init__(self):
        super().__init__()
        self.n_actions = len(self.action_map.keys())+1

        self.depots = dict([(color, Depot(color=color)) for color in ['red', 'blue', 'green', 'yellow']])
        self.depots['red'].position = (0,0)
        self.depots['yellow'].position = (4,0)
        self.depots['blue'].position = (4,3)
        self.depots['green'].position = (0,4)

        self.passenger = None
        self.passengers = [Passenger(name='Passenger')]
        self.goal = None
        self.reset()

    def reset(self):
        start_depots = list(self.depots.keys())
        random.shuffle(start_depots)
        for i,p in enumerate(self.passengers):
            p.position = self.depots[start_depots[i]].position
        self.agent.position = self.depots[start_depots[-1]].position

        goal_depots = list(self.depots.keys())
        random.shuffle(goal_depots)
        N = len(self.passengers)
        while all([g == s for g, s in zip(goal_depots[:N], start_depots[:N])]):
            random.shuffle(goal_depots)
        self.goal_dict = dict([(p.name, g) for p, g in zip(self.passengers, goal_depots[:N])])
        self.goal = TaxiGoal(self.goal_dict)

    def plot(self):
        ax = super().plot()
        for _, depot in self.depots.items():
            depot.plot(ax)
        for p in (p for p in self.passengers if not p.intaxi):
            p.plot(ax)
        for p in (p for p in self.passengers if p.intaxi):
            p.plot(ax)

        if self.goal:
            self.goal.plot()

    def step(self, action):
        if action < 4:
            super().step(action)
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

class TaxiOfHanoi(TaxiWorld):
    def __init__(self):
        super().__init__()
        self.passengers = [Passenger(name=name) for name in ['Alice', 'Bob', 'Carol']]
        self.reset()

class TaxiGoal(BasicGrid):
    def __init__(self, goal_dict):
        super().__init__(rows=1, cols=5)

        # Clear walls
        self._grid[:,:] = 0

        self.depots = dict([(color, Depot(color=color)) for color in ['red', 'blue', 'green', 'yellow']])
        self.depots['red'].position = (0,1)
        self.depots['yellow'].position = (0,2)
        self.depots['blue'].position = (0,3)
        self.depots['green'].position = (0,4)

        self.passengers = [Passenger(name=name) for name in list(goal_dict.keys())]
        for p in self.passengers:
            p.position = self.depots[goal_dict[p.name]].position

    def plot(self):
        ax = super().plot()
        for _, depot in self.depots.items():
            depot.plot(ax)
        for p in self.passengers:
            p.plot(ax)
        x, y = (0.5, 0.5)
        plt.text(0.5,0.5, 'Goal:', fontsize=12, color='k',
            horizontalalignment='center', verticalalignment='center')
