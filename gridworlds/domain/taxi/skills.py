import numpy as np
from .taxi import TaxiGrid5x5, TaxiGrid10x10
from ..gridworld.skills import GoToGridPosition

skills5x5 = ['interact'] + list(TaxiGrid5x5.depot_names)
skills10x10 = ['interact'] + list(TaxiGrid10x10.depot_names)

def run_skill(w, name):
    if name in w.depot_locs.keys():
        skill = lambda w, x: GoToDepot(w, x, name)
    else:
        skill = lambda w, x: Interact(w)
    while True:
        can_run, a, term = skill(w, w.agent.position)
        print(can_run, a, term)
        assert (can_run or term)
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
        raise ValueError('Invalid skill name' + str(skill_name))
    return skill(w, w.agent.position)

def GoToDepot(gridworld, start, depotname):
    depot = gridworld.depot_locs[depotname]
    return GoToGridPosition(gridworld, start=start, target=depot)[0]

def Interact(gridworld):
    # Check relevant state variables to see if skill can run
    agent_pos = gridworld.agent.position
    at_depot = any(np.all(loc == agent_pos) for _, loc in gridworld.depot_locs.items())
    at_passenger = any(np.all(p.position == agent_pos) for p in gridworld.passengers)
    crowded = (gridworld.passenger is not None and any(
        np.all(p.position == agent_pos) for p in gridworld.passengers if p != gridworld.passenger))

    if at_depot and at_passenger and not crowded:
        can_run = True
        action = 4
    else:
        # Nothing to do
        can_run = False
        action = None

    terminate = True  # skill always terminates in one step
    return (can_run, action, terminate)
