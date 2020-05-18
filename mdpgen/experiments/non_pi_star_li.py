import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP, random_reward_matrix
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, graph_value_fns


#%%
# This reproduces the example from Li, Walsh, Littman (2006) where a
# π*-preserving abstraction does not have the same optimal policy as the ground MDP
#
# Note that because B(x|z) depends on the action selected at s0, B is not Markov.
# Similarly, R(z',a,z) depends on the same additional history, so the abstraction
# is not Markov either.
T_list = np.array([
    [[0, 1, 0, 0.0],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 1]],

    [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 1]],
])
R = np.array([
    [0, 0.5, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 0, 0]
])
phi = np.array([
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [0,0,1],
])
mdp1 = MDP(T_list, [R, R], gamma=0.9)
mdp2 = AbstractMDP(mdp1, phi, p0=np.array([1,0,0,0]), t=1)
mdp2 = AbstractMDP(mdp1, phi)
is_markov(mdp2)

pi_g_list = mdp2.piecewise_constant_policies()
pi_a_list = mdp2.abstract_policies()

v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

order_v_g = np.stack(sort_value_fns(v_g_list)).round(4)
order_v_a = np.stack(sort_value_fns(v_a_list)).round(4)

mdp2.p0
agg_state = mdp2.phi.sum(axis=0) > 1
np.stack([mdp2.B(pi, t=1)[agg_state] for pi in pi_g_list])


v_phi_pi_phi_star, _, pi_phi_star = vi(mdp2)
v_pi_phi_star = vi(mdp1, mdp2.get_ground_policy(pi_phi_star))[0]

# Look for examples of v_pi_phi_star < v
for v in v_g_list:
    if compare_value_fns(v_pi_phi_star, v) == "<":
        print('Found example of order mismatch.')
        break
else:
    print('All examples had proper ordering.')
#%%
graph_value_fns(v_g_list)#, 'graphviz/non_markov_b/ground_17')
graph_value_fns(v_a_list)#, 'graphviz/non_markov_b/abstract_17')

v_pi_phi_star
np.asarray(v_g_list).round(3)
np.asarray(v_a_list).round(3)
