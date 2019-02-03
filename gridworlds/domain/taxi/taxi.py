import numpy as np
import matplotlib.pyplot as plt
import random
from ..gridworld.grid import BaseGrid
from ..gridworld.gridworld import GridWorld
from ..gridworld.objects.passenger import Passenger
from ..gridworld.objects.depot import Depot

class TaxiGrid5x5(BaseGrid):
    depot_locs = {
        'red':     (0,0),
        'yellow':  (4,0),
        'blue':    (4,3),
        'green':   (0,4),
    }
    depot_names = depot_locs.keys()
    def __init__(self):
        super().__init__(rows=5, cols=5)
        self._grid[1:4,4] = 1
        self._grid[7:10,2] = 1
        self._grid[7:10,6] = 1

class TaxiGrid10x10(BaseGrid):
    depot_locs = {
        'red':     (0,0),
        'blue':    (8,0),
        'green':   (9,4),
        'yellow':  (0,5),
        'gray':    (3,3),
        'magenta': (4,6),
        'cyan':    (0,8),
        'orange':  (9,9),
    }
    depot_names = depot_locs.keys()
    def __init__(self):
        super().__init__(rows=10, cols=10)
        self._grid[1:8,6] = 1
        self._grid[13:20,2] = 1
        self._grid[13:20,8] = 1
        self._grid[5:12,12] = 1
        self._grid[1:8,16] = 1
        self._grid[13:20,16] = 1

class BaseTaxi(GridWorld):
    def __init__(self):
        super().__init__()
        self.actions.append(4)# Add interact action
        self.passenger = None

    def reset(self, goal=True):
        # Place depots
        self.depots = dict()
        for name in self.depot_names:
            self.depots[name] = Depot(color=name)
            self.depots[name].position = self.depot_locs[name]

        # Place passengers and taxi
        start_depots = list(self.depot_names)
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
            self.passenger_goals = dict([(p.name, g) for p, g in zip(self.passengers, goal_depots[:N])])
            self.goal = TaxiGoal(self.passenger_goals)
        else:
            self.goal = None

    def plot(self, ax=None, goal_ax=None):
        ax = super().plot(ax)
        for _, depot in self.depots.items():
            depot.plot(ax)
        for p in (p for p in self.passengers if not p.intaxi):
            p.plot(ax)
        for p in (p for p in self.passengers if p.intaxi):
            p.plot(ax)

        if self.goal:
            self.goal.plot(goal_ax)

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
        s = self.get_state()
        if (self.goal is None) or not self.check_goal(s):
            done = False
        else:
            done = True
        r = -1.0 if not done else 1000
        return s, r, done

    def get_state(self):
        state = []
        x, y = self.agent.position
        state.extend([x, y])
        for p in self.passengers:
            x, y = p.position
            intaxi = p.intaxi
            state.extend([x, y, intaxi])
        return np.asarray(state, dtype=int)

    def get_goal_state(self):
        state = []
        for p in self.passengers:
            goal_name = self.passenger_goals[p.name]
            x, y = self.depots[goal_name].position
            intaxi = False
            state.extend([x, y, intaxi])
        return np.asarray(state, dtype=int)

    def check_goal(self, state):
        goal = self.get_goal_state()
        if np.all(state[2:]==goal):# ignore taxi, check passenger positions
            return True
        else:
            return False

class TaxiGoal(BaseGrid):
    def __init__(self, passenger_goals):
        super().__init__(rows=1, cols=1+len(passenger_goals))
        self._grid[:,:] = 0 # Clear walls

        colors = [color for passenger, color in passenger_goals.items()]
        self.depots = dict([(color, Depot(color=color)) for color in colors])
        for i, color in enumerate(colors):
            self.depots[color].position = (0,1+i)

        self.passengers = [Passenger(name=name) for name in list(passenger_goals.keys())]
        for p in self.passengers:
            p.position = self.depots[passenger_goals[p.name]].position

    def plot(self, ax):
        ax = super().plot(ax, draw_bg_grid=False)
        for _, depot in self.depots.items():
            depot.plot(ax)
        for p in self.passengers:
            p.plot(ax)
        ax.text(0,0.5, 'Goal:', fontsize=12, color='k',
            horizontalalignment='center', verticalalignment='center')
        return ax

class Taxi5x5(BaseTaxi, TaxiGrid5x5):
    name = 'Taxi5x5'
    def __init__(self):
        super().__init__()
        self.passengers = [Passenger(name='Passenger')]
        self.reset()

class BusyTaxi5x5(BaseTaxi, TaxiGrid5x5):
    name = 'BusyTaxi5x5'
    def __init__(self):
        super().__init__()
        self.passengers = [Passenger(name=name) for name in ['Alice', 'Bob', 'Carol']]
        self.reset()

class Taxi10x10(BaseTaxi, TaxiGrid10x10):
    name = 'Taxi10x10'
    def __init__(self):
        super().__init__()
        self.passengers = [Passenger(name='Passenger')]
        self.reset()

class BusyTaxi10x10(BaseTaxi, TaxiGrid10x10):
    name = 'BusyTaxi10x10'
    def __init__(self):
        super().__init__()
        names = ['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'George']
        self.passengers = [Passenger(name=name) for name in names]
        self.reset()
