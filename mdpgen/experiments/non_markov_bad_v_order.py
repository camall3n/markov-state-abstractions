import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, graph_value_fns

#%%
# This illustrates an example where V^{\pi_\phi^*} < max_{\pi\in \Pi_\phi} V^{\pi}
# Note the fixed weighting scheme.
T_list = np.array([
    [[1., 0., 0.],
     [1., 0., 0.],
     [0., 0., 1.]],

    [[0., 1., 0.],
     [0., 0., 1.],
     [0., 1., 0.]]
])
R_list = np.array([
   [[1.,  0., 0.],
    [0.5, 0., 0.],
    [0.,  0., 0.5]],

   [[0., 1.,  0.],
    [0., 0.,  1.],
    [0., 0.1, 0.]]
])
phi = np.array([
    [0, 1],
    [1, 0],
    [0, 1]
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

# Look for examples of v_pi_phi_star < v
for v in v_g_list:
    if compare_value_fns(v_pi_phi_star, v) == "<":
        break
else:
    print('No examples found.')

#%%
graph_value_fns(v_g_list)
graph_value_fns(v_a_list)

v_pi_phi_star
np.asarray(v_g_list).round(3)
np.asarray(v_a_list).round(3)


#%%
# This illustrates an example where V^{\pi_\phi^*} < max_{\pi\in \Pi_\phi} V^{\pi}
# Note the fixed weighting scheme.
T_list = np.array([
    [[1., 0., 0.],
     [1., 0., 0.],
     [0., 0., 1.]],

    [[0., 1., 0.],
     [0., 0., 1.],
     [0., 1., 0.]]
])
R_list = np.array([
   [[1.,  0., 0.],
    [0.5, 0., 0.],
    [0.,  0., 0.5]],

   [[0., 1.,  0.],
    [0., 0.,  1.],
    [0., 0.1, 0.]]
])
phi = np.array([
    [0, 1],
    [1, 0],
    [0, 1]
])

mdp1 = MDP(T_list, R_list, gamma=0.9)
mdp2 = AbstractMDP(mdp1, phi)

pi_g_list = mdp2.piecewise_constant_policies()
pi_a_list = mdp2.abstract_policies()

v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

order_v_g = sort_value_fns(v_g_list)
order_v_a = sort_value_fns(v_a_list)

v_phi_pi_phi_star, _, pi_phi_star = vi(mdp2)
v_pi_phi_star = vi(mdp1, mdp2.get_ground_policy(pi_phi_star))[0]

# Look for examples of v_pi_phi_star < v
for v in v_g_list:
    if compare_value_fns(v_pi_phi_star, v) == "<":
        break
else:
    print('No examples found.')

#%%
graph_value_fns(v_g_list, 'bad_v_order_gnd')
graph_value_fns(v_a_list, 'bad_v_order_abs')


v_pi_phi_star
pi_g_list
pi_a_list
np.asarray(v_g_list).round(2)
np.asarray(v_a_list).round(2)
