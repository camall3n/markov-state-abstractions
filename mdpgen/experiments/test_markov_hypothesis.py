import gmpy
import numpy as np
from tqdm import tqdm

from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP
from mdpgen.vi import vi
from mdpgen.markov import generate_markov_mdp_pair, generate_non_markov_mdp_pair, is_markov

from mdpgen.value_fn import compare_value_fns, partial_ordering, sorted_order, sort_value_fns, graph_value_fns

#%%
for _ in tqdm(range(100)):
    # Generate (MDP, abstract MDP) pair
    # mdp1, mdp2 = generate_non_markov_mdp_pair(
    #     n_states=3, n_abs_states=2, n_actions=2, sparsity=1,
    #     fixed_w=True,
    # )
    mdp1, mdp2 = generate_markov_mdp_pair(
        n_states=5, n_abs_states=3, n_actions=2, sparsity=.7,
        equal_block_rewards = True,
        equal_block_transitions = True,
    )
    mdp2.phi
    is_markov(mdp2)

    pi_g_list = mdp2.piecewise_constant_policies()
    pi_a_list = mdp2.abstract_policies()

    # B_list = [AbstractMDP(mdp1, mdp2.phi, p0=np.array([1,0,0,0])).B(pi, t=1).round(3) for pi in pi_g_list]
    # i=5
    # pi_gnd = pi_g_list[i]
    # pi_abs = pi_a_list[i]
    # mdp2.B(pi_gnd)
    # phi = mdp2.phi
    # N_gnd = mdp1.get_N(pi_gnd)
    # phi.transpose() @ phi
    # Px = mdp1.stationary_distribution(pi_gnd)
    # N_abs = mdp2.get_N(pi_abs)
    # Pz = mdp2.stationary_distribution(pi_abs)
    #
    # ratio_abs = np.divide(N_abs, Pz[None,:], out=np.zeros_like(N_abs), where=Pz!=0)
    # ratio_gnd = np.divide(N_gnd, Px[None,:], out=np.zeros_like(N_gnd), where=Px!=0)
    # mdp2.B(pi_gnd) @ ratio_gnd
    # ratio_abs @ phi.transpose()
    # is_markov(mdp2)

    v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
    v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

    order_v_g = sort_value_fns(v_g_list)
    order_v_a = sort_value_fns(v_a_list)

    agg_state = mdp2.phi.sum(axis=0) > 1
    [mdp2.B(pi, t=0)[agg_state][0] for pi in pi_g_list]
    [mdp2.B(pi, t=1)[agg_state][0] for pi in pi_g_list]
    [mdp2.B(pi, t=3)[agg_state][0] for pi in pi_g_list]

    v_star, _, pi_star = vi(mdp1)
    v_phi_pi_phi_star, _, pi_phi_star = vi(mdp2)
    pi_phi_star_gnd = mdp2.get_ground_policy(pi_phi_star)
    v_pi_phi_star = vi(mdp1, pi_phi_star_gnd)[0]

    for v in v_g_list:
        if compare_value_fns(v_pi_phi_star, v) == "<":
            print('Inconsistent value ordering')
            break
    else:
        if compare_value_fns(v_pi_phi_star, v_star) == "<":
            print('Found example that was non π*-preserving.')
            break
        continue
    break
else:
    print('No examples found.')

# #%%
graph_value_fns(v_g_list)
graph_value_fns(v_a_list)

#%%
v_pi_phi_star
np.asarray(sort_value_fns(v_g_list)).round(3)

#%%
mdp3, mdp4 = generate_markov_mdp_pair(
    n_states=3, n_abs_states=2, n_actions=2, sparsity=1,
    equal_block_rewards = False,
    equal_block_transitions = False,
)
