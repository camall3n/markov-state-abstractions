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
    mdp1, mdp2 = generate_markov_mdp_pair(
        n_states=4, n_abs_states=3, n_actions=2,
        equal_block_rewards=False,
        equal_block_transitions=False,
    )

    v_star, q_star, pi_star = vi(mdp1)
    v_star, pi_star

    pi_g_list = mdp2.piecewise_constant_policies()
    pi_a_list = mdp2.abstract_policies()
    v_g_list = [vi(mdp1, pi)[0] for pi in pi_g_list]
    v_a_list = [vi(mdp2, pi)[0] for pi in pi_a_list]

    order_v_g = sorted_order(v_g_list)
    order_v_a = sorted_order(v_a_list)
    assert np.allclose(order_v_a, order_v_g)
print('All tests passed.')

#%%
graph_value_fns(v_g_list, 'graphviz/arbitrary_both/ground_10')
graph_value_fns(v_a_list, 'graphviz/arbitrary_both/abstract_10')


#%%
v_phi_star, q_phi_star, pi_phi_star = vi(mdp2)
v_phi_star

n_policies = mdp1.n_actions**mdp1.n_states
def get_policy(mdp, i):
    assert i < n_policies
    pi_string = gmpy.digits(i, mdp.n_actions).zfill(mdp.n_states)
    pi = np.asarray(list(pi_string), dtype=int)
    return pi

# for each ground-state policy
#%%
n_policies = mdp1.n_actions**mdp1.n_states
for i in range(n_policies):
    pi = get_policy(mdp1, i)

    # compare V^pi vs V_phi^pi
    v_pi = vi(mdp1, pi)[0]
    belief = mdp2.B(pi)
    v_phi_pi = belief @ v_pi
    print(i, pi)
    print(v_pi)
    print(v_phi_pi)
    print()

#%%
# for each pair of ground-state policies
pi_list = []
v_list = []
v_phi_list = []
for i in range(n_policies):
    pi = get_policy(mdp1, i)

    v = vi(mdp1, pi)[0]

    belief = mdp2.B(pi)
    v_phi = belief @ v

    pi_list.append(pi)
    v_list.append(v)
    v_phi_list.append(v_phi)

#%%
for pi_i, v_i, v_phi_i in zip(pi_list, v_list, v_phi_list):
    for pi_j, v_j, v_phi_j in zip(pi_list, v_list, v_phi_list):
        if np.all(pi_j==pi_i):
            continue

        # check which values are close to equal in pi_i and pi_j
        z_eq_mask = ~np.isclose(v_phi_i,v_phi_j, atol=1e-4)
        x_eq_mask = ~np.isclose(v_i,v_j,atol=1e-4)
        # such values should be close for all ground states in those abstract states
        if not np.all(((x_eq_mask @ phi)>0) == z_eq_mask):
            print('incompatible masks??')
            break

        # if so, check which remaining state values are greater in pi_i than pi_j
        v_phi_gt = v_phi_i[z_eq_mask] > v_phi_j[z_eq_mask]
        v_gt = v_i[x_eq_mask] > v_j[x_eq_mask]

        # and check which are less
        v_phi_lt = v_phi_i[z_eq_mask] < v_phi_j[z_eq_mask]
        v_lt = v_i[x_eq_mask] < v_j[x_eq_mask]

        # if all ground values are greater (or lesser) in pi_i than pi_j but
        # not all abstract values are, then we might have a problem
        # otherwise, maybe the hypothesis is right...?
        if (np.all(v_gt) and not np.all(v_phi_gt)) or (np.all(v_lt) and not np.all(v_phi_lt)):
            print('hypothesis might be wrong!')
            print(pi_star)
            print(pi_phi_star @ phi.transpose().astype(int))
            break

        # what if there's a mix of greater/lesser values?
        # e.g. V_pi1(x1) > V_pi2(x1), but V_pi1(x2) < V_pi2(x2)

    else:
        continue
    break
else:
    print('hypothesis could be correct!')

    # compare V^pi vs V_phi^pi
    # print(i, pi, v_pi, v_phi_pi)

# (mdp.T[0] * mdp.R[0]).sum(axis=1)
# (mdp.T[1] * mdp.R[1]).sum(axis=1)
#%%
print(i, pi_i)
print(v_i)
print(v_phi_i)
print()
print(j, pi_j)
print(v_j)
print(v_phi_j)
print()
