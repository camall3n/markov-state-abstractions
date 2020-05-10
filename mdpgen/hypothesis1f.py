import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, visualize_order


#%%
for _ in tqdm(range(100)):
    # Generate (MDP, abstract MDP) pair
    # mdp1, mdp2 = generate_non_markov_mdp_pair(
    #     n_states=3, n_abs_states=2, n_actions=2,
    #     fixed_w=True,
    # )
    mdp1, mdp2 = generate_markov_mdp_pair(
        n_states=3, n_abs_states=2, n_actions=2,
        equal_block_rewards = False,
        equal_block_transitions = False,
    )

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
        continue
    break
else:
    print('No examples found.')

# #%%
visualize_order(v_g_list, 'graphviz/non_markov/ground_14')
visualize_order(v_a_list, 'graphviz/non_markov/abstract_14')

#%%
v_pi_phi_star
np.asarray(sort_value_fns(v_g_list)).round(3)
