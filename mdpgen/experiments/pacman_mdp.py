import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP, random_reward_matrix
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov, is_hutter_markov, has_block_dynamics

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, graph_value_fns


#%%
# This illustrates an example where V^{\pi_\phi^*} < max_{\pi\in \Pi_\phi} V^{\pi}
# In this case, the abstraction induces a non-markov belief distribution
T_list = np.array([
    [[0, 0.4, 0.6, 0, 0],
     [0, 0.,  0.,  1, 0],
     [0, 0.,  0.,  0, 1],
     [1, 0.,  0.,  0, 0],
     [1, 0.,  0.,  0, 0]],

    [[0, 0, 0, 0.4, 0.6],
     [1, 0, 0, 0.,  0.],
     [1, 0, 0, 0.,  0.],
     [0, 1, 0, 0.,  0.],
     [0, 0, 1, 0.,  0.]],
])
equal_block_rewards = True
markov_abstraction = True
for i in tqdm(range(20)):
    R_list = random_reward_matrix(0, 1, T_list.shape) * (T_list > 0)
    # If we constrain the rewards so they have block structure, everything
    # is fine. Otherwise it's easy to find examples where the policy ranking
    # changes badly
    if equal_block_rewards:
        R_list[0,3:,0] = R_list[0,3:,0].mean()
        R_list[0,0,1:3] = R_list[0,0,1:3].mean()
        R_list[0,1:3,3:5] = R_list[0,1:3,3:5].sum()/2
        R_list[1,0,3:] = R_list[1,0,3:].mean()
        R_list[1,1:3,0] = R_list[1,1:3,0].mean()
        R_list[1,3:5,1:3] = R_list[1,3:5,1:3].sum()/2
    if not markov_abstraction:
        phi = np.array([
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 1]
        ])
    else:
        # However, if we enforce a Markov abstraction, the policy
        # ranking remains good even for arbitrary rewards
        phi = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ])

    mdp1 = MDP(T_list, R_list, gamma=0.9)
    mdp2 = AbstractMDP(mdp1, phi, p0=np.array([0,0,0,1,0]), t=200)
    mdp2 = AbstractMDP(mdp1, phi)
    is_markov(mdp2)
    is_hutter_markov(mdp2)
    has_block_dynamics(mdp2)

    pi_g_list = mdp2.piecewise_constant_policies()
    pi_a_list = mdp2.abstract_policies()

    v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
    v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

    order_v_g = np.stack(sort_value_fns(v_g_list)).round(4)
    order_v_a = np.stack(sort_value_fns(v_a_list)).round(4)

    mdp2.p0
    agg_state = mdp2.phi.sum(axis=0) > 1
    np.stack([mdp2.B(pi, t=6)[agg_state] for pi in pi_g_list])


    v_star, _, pi_star = vi(mdp1)
    v_phi_pi_phi_star, _, pi_phi_star = vi(mdp2)
    v_pi_phi_star = vi(mdp1, mdp2.get_ground_policy(pi_phi_star))[0]

    # Look for examples of v_pi_phi_star < v
    for v in v_g_list:
        if compare_value_fns(v_pi_phi_star, v) == "<":
            print('Found example of order mismatch.')
            break
    else:
        # if compare_value_fns(v_pi_phi_star, v_star) == "<":
        #     print('Found example that was non π*-preserving.')
        #     break
        continue
    break
else:
    print('All examples had proper ordering.')
#%%
graph_value_fns(v_g_list)#, 'graphviz/non_markov_b/ground_17')
graph_value_fns(v_a_list)#, 'graphviz/non_markov_b/abstract_17')

v_pi_phi_star
np.asarray(v_g_list).round(3)
np.asarray(v_a_list).round(3)
