from .taxi import TaxiGrid5x5, TaxiGrid10x10
from ..gridworld.skills import GoToGridPosition

skills5x5 = ['interact']+list(TaxiGrid5x5.depot_names)
skills10x10 = ['interact']+list(TaxiGrid10x10.depot_names)
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
    return GoToGridPosition(gridworld, start=start, target=depot)

def Interact(gridworld):
    base_action = 4
    s0 = gridworld.get_state()
    gridworld.step(base_action)# try interacting
    s1 = gridworld.get_state()
    gridworld.step(base_action)# undo

    # Can only run if it would have an effect
    can_run = False if all(s0 == s1) else True
    action = base_action
    terminate = True
    return (can_run, action, terminate)
