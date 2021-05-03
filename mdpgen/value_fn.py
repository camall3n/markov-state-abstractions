from collections import defaultdict
import itertools

from graphviz import Digraph
import numpy as np
# from toposort import toposort

from mdpgen.mdp import MDP
from mdpgen.vi import vi

def compare_value_fns(v1, v2):
    # Since >= doesn't make sense for floating point, we have to check for
    # (near) equality first, and then check the remaining indices
    eq_idx = np.isclose(v1, v2)
    if np.all(eq_idx):
        return '='
    gt_idx = v1[~eq_idx] > v2[~eq_idx]
    lt_idx = v1[~eq_idx] < v2[~eq_idx]
    if not np.any(gt_idx):
        return '<'
    elif not np.any(lt_idx):
        return '>'
    else:
        return '?'

def graph_value_fns(v_list, filename=None):
    dot = Digraph()
    for i in range(len(v_list)):
        dot.node('π_'+str(i), 'π_'+str(i))
    for i, v_i in enumerate(v_list):
        for j, v_j in enumerate(v_list):
            if compare_value_fns(v_i, v_j) == '<':
                dot.edge('π_'+str(j),'π_'+str(i))
    if filename is not None:
        dot.render(filename, format='png', view=True)
    return dot

def preference_map(v_list):
    # pref[a] = {b, c} means a is preferred to b or c
    prefs = defaultdict(set)
    for i, v_i in enumerate(v_list):
        for j, v_j in enumerate(v_list):
            if compare_value_fns(v_i, v_j) == '>':
                prefs[i].add(j)
    return prefs

def partial_ordering(v_list):
    # build dependency graph for topological sort
    # dependency(a,b) means v_list[a] < v_list[b]
    dependencies = defaultdict(set)
    for i, v_i in enumerate(v_list):
        dependencies[i].add(i)
        for j, v_j in enumerate(v_list):
            if compare_value_fns(v_i, v_j) == '<':
                dependencies[i].add(j)
    return list(toposort(dependencies))

def sorted_order(v_list):
    sorted_idxs = partial_ordering(v_list)
    # sorted_idxs is a list of sets.
    # label each item of each with its set's position in the list
    positions = [[(element, i) for element in set_i] for (i, set_i) in enumerate(sorted_idxs)]
    # then flatten to determine the final ordering
    ordered_idxs = list(itertools.chain.from_iterable(map(sorted, positions)))

    # ignore the position; return only the rank
    return [order for (order, pos) in ordered_idxs]

def sort_value_fns(v_list):
    order = sorted_order(v_list)
    return np.stack(v_list)[order].tolist()

#%%
def test():
    mdp = MDP.generate(n_states=4, n_actions=2)
    pi_list = mdp.all_policies()
    v_list = [vi(mdp, pi)[0] for pi in pi_list]
    v_ranks = sorted_order(v_list)

    sorted_v = [v for _,v in sorted(zip(v_ranks, v_list))]
    for v1, v2 in zip(sorted_v[:-1], sorted_v[1:]):
        assert compare_value_fns(v1, v2) != '<'
    # for pi1, v1 in zip(pi_list, v_list):
    #     for pi2, v2 in zip(pi_list, v_list):
    #         print(v1.round(4))
    #         print(compare_value_fns(v1, v2), v2.round(4))
    #         print()

    v_star, _, pi_star = vi(mdp)
    assert compare_value_fns(v_star, sorted_v[0]) == '='

def test2():
    v_list1 = [
        np.array([-0.93322951, -0.71563646, -1.09864307, -1.41972321]),
        np.array([0.28745757, 0.54022545, 0.28496826, 0.66206808]),
        np.array([-1.08895623, -0.85832019, -1.31753513, -1.57121617]),
        np.array([0.09340452, 0.36117643, 0.01522051, 0.46515426]),
        np.array([-1.75472834, -2.31375144, -1.79897509, -2.15814589]),
        np.array([-0.34170253, -0.6454437 , -0.26313392,  0.06281763]),
        np.array([-2.06654728, -2.59899574, -2.23751442, -2.46152071]),
        np.array([-0.6574191 , -0.94002037, -0.70023168, -0.25734133]),
        np.array([ 0.32409196,  0.3268805 , -0.09826439, -0.45996648]),
        np.array([1.94867749, 1.92454514, 1.61841041, 1.96284172]),
        np.array([ 0.25456239,  0.25601484, -0.24615639, -0.54482959]),
        np.array([1.85538073, 1.82986457, 1.43219225, 1.8458723 ]),
        np.array([-1.10071811, -1.86377632, -1.28331626, -1.67012266]),
        np.array([0.73880886, 0.10753569, 0.59459354, 0.88513743]),
        np.array([-1.28034793, -2.06184239, -1.62791659, -1.87862925]),
        np.array([ 0.53763429, -0.11345431,  0.23281229,  0.64546823])
    ]
    v_list2 = [
        np.array([-0.75738645, -0.7156197 , -1.41970646, -1.09862631, -1.10531084]),
        np.array([-0.91320879, -0.85830596, -1.57120194, -1.3175209 , -1.26094897]),
        np.array([0.51329831, 0.54022545, 0.66206808, 0.28496826, 0.06640554]),
        np.array([ 0.31868082,  0.36117643,  0.46515426,  0.01522051, -0.12709503]),
        np.array([-1.49525435, -2.31375144, -2.15814589, -1.79897509, -2.00870046]),
        np.array([-1.80730539, -2.59899574, -2.46152071, -2.23751442, -2.32029222]),
        np.array([-0.05775841, -0.64543801,  0.06282331, -0.26312824, -0.61961466]),
        np.array([-0.37407516, -0.94002037, -0.25734133, -0.70023168, -0.93475504]),
        np.array([ 0.55089574,  0.32687724, -0.45996974, -0.09826765,  0.10209086]),
        np.array([ 0.48136431,  0.2560135 , -0.54483093, -0.24615773,  0.03256691]),
        np.array([2.06627248, 1.92454514, 1.96284172, 1.61841041, 1.83357597]),
        np.array([1.9737398 , 1.82986457, 1.8458723 , 1.43219225, 1.73953134]),
        np.array([-0.87063868, -1.86376735, -1.67011369, -1.28330729, -1.3259012 ]),
        np.array([-1.05020869, -2.06183164, -1.8786185 , -1.62790583, -1.50558604]),
        np.array([0.8642462 , 0.10753569, 0.88513743, 0.59459354, 0.61603127]),
        np.array([ 0.66466002, -0.11345431,  0.64546823,  0.23281229,  0.413302  ])
    ]
    v_list1[9]
    v_list1[11]
    sorted_order(v_list1)
    sorted_order(v_list2)
    sort_value_fns(v_list1)
    sort_value_fns(v_list2)

    print('All tests passed.')

if __name__ == '__main__':
    test()
