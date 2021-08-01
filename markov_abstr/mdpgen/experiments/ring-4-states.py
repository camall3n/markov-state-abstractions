import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, graph_value_fns


#%%
# Generate (MDP, abstract MDP) pair
T = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
])
R = np.array([
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [2, 2, 2, 2],
    [1, 1, 1, 1],
])/4

mdp1 = MDP([T, T.transpose()], [R, R.transpose()], gamma=0.9)
phi = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
])
mdp2 = UniformAbstractMDP(mdp1, phi)
is_markov(mdp2)

v_g_star, q_g_star, pi_g_star = vi(mdp1)
v_g_star, pi_g_star

v_a_star, q_a_star, pi_a_star = vi(mdp2)
v_a_star, pi_a_star


pi_g_list = mdp2.piecewise_constant_policies()
pi_a_list = mdp2.abstract_policies()
v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

order_v_g = sorted_order(v_g_list)
order_v_a = sorted_order(v_a_list)
print(partial_ordering(v_g_list))
print(partial_ordering(v_a_list))
assert np.allclose(order_v_a, order_v_g)

#%%
graph_value_fns(v_g_list, 'graphviz/non_markov/ground_11')
graph_value_fns(v_a_list, 'graphviz/non_markov/abstract_11')
