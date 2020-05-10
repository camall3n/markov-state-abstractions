import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, visualize_order

# This illustrates an example where V^{\pi_\phi^*} < max_{\pi\in \Pi_\phi} V^{\pi}
# Note the fixed weighting scheme.

#%%
T_list = [
    np.array([
        [0.27776928, 0.57841312, 0.1438176 ],
        [0.12747357, 0.23511909, 0.63740734],
        [0.35827644, 0.24496315, 0.39676041],
    ]),
    np.array([
        [0.35896708, 0.59838564, 0.04264727],
         [0.45729479, 0.11493169, 0.42777352],
         [0.37912883, 0.43052522, 0.19034595],
    ])
]
R_list = [
    np.array([
        [ 0.39, -0.54, -0.25],
        [ 0.86, -0.68,  0.41],
        [ 0.56, -0.98,  0.02],
    ]),
    np.array([
        [ 0.61,  0.7 ,  0.17],
        [ 0.03,  0.67,  0.14],
        [ 0.23, -0.8 ,  0.42],
    ])
]
phi = np.array([
    [0., 1.],
    [1., 0.],
    [1., 0.],
])

mdp1 = MDP(T_list, R_list, gamma=0.9)
mdp2 = UniformAbstractMDP(mdp1, phi)


pi_g_list = mdp2.piecewise_constant_policies()
pi_a_list = mdp2.abstract_policies()

v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

order_v_g = sort_value_fns(v_g_list)
order_v_a = sort_value_fns(v_a_list)

v_phi_pi_phi_star, _, pi_phi_star = vi(mdp2)
v_pi_phi_star = vi(mdp1, mdp2.get_ground_policy(pi_phi_star))[0]

for v in v_g_list:
    if compare_value_fns(v_pi_phi_star, v) == "<":
        break
else:
    print('No examples found.')

#%%
visualize_order(v_g_list, 'graphviz/non_markov/ground_14')
visualize_order(v_a_list, 'graphviz/non_markov/abstract_14')

#%%
v_pi_phi_star
np.asarray(sort_value_fns(v_g_list)).round(3)
