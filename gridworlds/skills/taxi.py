from collections import defaultdict
import numpy as np
from ..domain.taxi import TaxiDomain5x5, TaxiDomain10x10
from ..utils import manhattan_dist
from ..grid.basicgrid import directions, actions

skills5x5 = ['interact']+list(TaxiDomain5x5.depot_names)
skills10x10 = ['interact']+list(TaxiDomain10x10.depot_names)
def run_skill(w, name):
    if name in w.depot_locs.keys():
        skill = lambda w, x: GoToDepot(w, x, name)
    else:
        skill = lambda w, x: Interact(w)
    while True:
        can_run, a, term = skill(w, w.agent.position)
        print(can_run, a, term)
        assert(can_run or term)
        if can_run:
            w.step(a)
        if term:
            break
def skill_policy(w, skill_name):
    if skill_name in w.depot_locs.keys():
        skill = lambda w, x: GoToDepot(w, x, skill_name)
    elif skill_name == 'interact':
        skill = lambda w, x: Interact(w)
    else:
        raise ValueError('Invalid skill name'+str(skill_name))
    return skill(w, w.agent.position)


def GoToDepot(gridworld, start, depotname):
    depot = gridworld.depot_locs[depotname]
    action = GoToGridPosition(gridworld, start=start, target=depot)
    can_run = True if action is not None else False
    terminate = True if all(start == depot) else False
    return (can_run, action, terminate)

def Interact(gridworld):
    s0 = gridworld.get_state()
    gridworld.step(4)# try interacting
    s1 = gridworld.get_state()
    gridworld.step(4)# undo

    # Can only run if it would have an effect
    can_run = False if all(s0 == s1) else True
    action = 4
    terminate = True
    return (can_run, action, terminate)

def GoToGridPosition(grid, start, target):
    start = tuple(start)
    target = tuple(target)
    # Cache results to save on repeated calls
    if (start, target) not in grid.saved_directions:
        path = GridAStar(grid, start, target)
        if path:
            for next, current in zip(path[:-1], path[1:]):
                direction = tuple(np.asarray(next)-current)
                grid.saved_directions[(current, target)] = direction
    direction = grid.saved_directions.get((start, target), None)
    action = actions[direction] if direction else None
    return action

def GridAStar(grid, start, target):
    # Use A* to search for a path in grid from start to target
    closed_set = set()
    open_set = set()
    came_from = dict()
    gScore = defaultdict(lambda x: np.inf)
    fScore = defaultdict(lambda x: np.inf)

    open_set.add(tuple(start))
    gScore[start] = 0
    fScore[start] = manhattan_dist(start, target)

    while open_set:
        current = min(open_set, key=lambda x: fScore[x])
        if all(np.asarray(current) == target):
            return _reconstruct_path(came_from, current)

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in get_neighbors(grid, current):
            if neighbor in closed_set:
                continue

            tentative_gScore = gScore[current] + 1

            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_gScore >= gScore[neighbor]:
                continue

            came_from[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + manhattan_dist(neighbor, target)

    return None

def get_neighbors(grid, pos):
    neighbors = []
    for _, dir in directions.items():
        if not grid.has_wall(pos, dir):
            neighbor = tuple(np.asarray(pos)+dir)
            neighbors.append(neighbor)
    return neighbors

def _reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path
