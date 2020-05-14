import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, graph_value_fns


#%%
# Try MDP pair with non-Markov belief
T = np.array([
    [0, .5, .5, 0, 0, 0],
    [0, 0, 0, .5, .5, 0],
    [0, 0, 0, 0, .5, .5],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
])
R = np.array([
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 2, 2],
    [2, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0]
])/4

mdp1 = MDP([T, T], [R, R], gamma=0.9)
phi = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])
mdp2 = AbstractMDP(mdp1, phi)

v_star, q_star, pi_star = vi(mdp1)
v_star, pi_star

pi_g_list = mdp2.piecewise_constant_policies()
pi_a_list = mdp2.abstract_policies()
v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

np.allclose(v_g_list, v_g_list[0])

order_v_g = sorted_order(v_g_list)
order_v_a = sorted_order(v_a_list)
assert np.allclose(order_v_a, order_v_g)

graph_value_fns(v_a_list)
graph_value_fns(v_g_list)
