from collections import defaultdict
import numpy as np
from ...utils import manhattan_dist
from .grid import directions, actions

def GoToGridPosition(gridworld, start, target):
    start = tuple(start)
    target = tuple(target)
    # Cache results to save on repeated calls
    if (start, target) not in gridworld.saved_directions:
        path = _GridAStarPath(gridworld, start, target)
        if path:
            for next, current in zip(path[:-1], path[1:]):
                direction = tuple(np.asarray(next)-current)
                gridworld.saved_directions[(current, target)] = direction
    direction = gridworld.saved_directions.get((start, target), None)
    action = actions[direction] if direction else None
    can_run = True if action is not None else False
    terminate = True if (start == target) else False
    return (can_run, action, terminate)

def _GridAStarPath(gridworld, start, target):
    # Use A* to search for a path in gridworld from start to target
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

        for neighbor in _get_neighbors(gridworld, current):
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

def _get_neighbors(gridworld, pos):
    neighbors = []
    for _, dir in directions.items():
        if not gridworld.has_wall(pos, dir):
            neighbor = tuple(np.asarray(pos)+dir)
            neighbors.append(neighbor)
    return neighbors

def _reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path
