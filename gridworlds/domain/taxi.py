import numpy as np
import matplotlib.pyplot as plt
import random
from ..grid.basicgrid import BasicGrid
from ..grid.taxigrid import TaxiGrid5x5, TaxiGrid10x10
from .gridworld import GridWorld
from ..objects.passenger import Passenger
from ..objects.depot import Depot

class BaseTaxi(GridWorld):
    def __init__(self):
        super().__init__()
        self.n_actions = len(self.action_map.keys())+1
        self.passenger = None

    def reset(self, goal=True):
        # Place depots
        self.depots = dict()
        for color, position in self.depot_locs.items():
            self.depots[color] = Depot(color=color)
            self.depots[color].position = position

        # Place passengers and taxi
        start_depots = list(self.depots.keys())
        random.shuffle(start_depots)
        for i,p in enumerate(self.passengers):
            p.position = self.depots[start_depots[i]].position
        self.agent.position = self.depots[start_depots[-1]].position

        if goal:
            # Generate goal condition
            goal_depots = list(self.depots.keys())
            random.shuffle(goal_depots)
            N = len(self.passengers)
            while all([g == s for g, s in zip(goal_depots[:N], start_depots[:N])]):
                random.shuffle(goal_depots)
            self.goal_dict = dict([(p.name, g) for p, g in zip(self.passengers, goal_depots[:N])])
            self.goal = TaxiGoal(self.goal_dict)
        else:
            self.goal = None

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
                # pick up?
                for p in self.passengers:
                    if (self.agent.position == p.position).all():
                        p.intaxi = True
                        self.passenger = p
                        break # max one passenger per taxi
            else:
                # drop off?
                dropoff_clear = True
                for p in (p for p in self.passengers if p is not self.passenger):
                    if (p.position == self.passenger.position).all():
                        dropoff_clear = False
                        break
                if dropoff_clear:
                    self.passenger.intaxi = False
                    self.passenger = None

    def get_state(self):
        state = []
        x, y = self.agent.position
        state.extend([x, y])
        for p in self.passengers:
            x, y = p.position
            intaxi = p.intaxi
            state.extend([x, y, intaxi])
        return state

    def get_goal_state(self):
        state = []
        for p in self.passengers:
            goal_depotname = self.goal_dict[p.name]
            x, y = self.depots[goal_depotname].position
            intaxi = False
            state.extend([x, y, intaxi])
        return state

    def goal_oracle(self, state):
        goal = self.get_goal_state()
        if all([s == g for s, g in zip(state[2:], goal)]):# ignore taxi, check passenger positions
            return True
        else:
            return False

class TaxiGoal(BasicGrid):
    def __init__(self, goal_dict):
        super().__init__(rows=1, cols=1+len(goal_dict))
        self._grid[:,:] = 0 # Clear walls

        colors = [color for passenger, color in goal_dict.items()]

        self.depots = dict([(color, Depot(color=color)) for color in colors])
        for i, color in enumerate(colors):
            self.depots[color].position = (0,1+i)

        self.passengers = [Passenger(name=name) for name in list(goal_dict.keys())]
        for p in self.passengers:
            p.position = self.depots[goal_dict[p.name]].position

    def plot(self):
        ax = super().plot()
        for _, depot in self.depots.items():
            depot.plot(ax)
        for p in self.passengers:
            p.plot(ax)
        plt.text(0.5,0.5, 'Goal:', fontsize=12, color='k',
            horizontalalignment='center', verticalalignment='center')


class TaxiDomain5x5(BaseTaxi, TaxiGrid5x5):
    def __init__(self):
        super().__init__()
        self.depot_locs = {
            'red':     (0,0),
            'yellow':  (4,0),
            'blue':    (4,3),
            'green':   (0,4),
        }

class TaxiDomain10x10(BaseTaxi, TaxiGrid10x10):
    def __init__(self):
        super().__init__()
        self.depot_locs = {
            'red':     (0,0),
            'blue':    (8,0),
            'green':   (9,4),
            'yellow':  (0,5),
            'gray':    (3,3),
            'magenta': (4,6),
            'cyan':    (0,8),
            'orange':  (9,9),
        }

class TaxiClassic(TaxiDomain5x5):
    def __init__(self):
        super().__init__()
        self.passengers = [Passenger(name='Passenger')]
        self.reset()

class BusyTaxi(TaxiDomain5x5):
    def __init__(self):
        super().__init__()
        self.passengers = [Passenger(name=name) for name in ['Alice', 'Bob', 'Carol']]
        self.reset()

class Taxi10x10(TaxiDomain10x10):
    def __init__(self):
        super().__init__()
        self.passengers = [Passenger(name='Passenger')]
        self.reset()

class BusyTaxi10x10(TaxiDomain10x10):
    def __init__(self):
        super().__init__()
        names = ['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'George']
        self.passengers = [Passenger(name=name) for name in names]
        self.reset()
