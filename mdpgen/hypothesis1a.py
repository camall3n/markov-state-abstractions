import gmpy
import numpy as np

from mdpgen.mdp import MDP, AbstractMDP, one_hot, random_sparse_mask, random_transition_matrix, is_stochastic
from mdpgen.vi import vi
from mdpgen.markov import is_markov

#%%
def random_phi(n_abs_states):
    phi = np.eye(n_abs_states)
    phi = np.concatenate((phi, random_sparse_mask((1,n_abs_states), sparsity=1)))
    np.random.shuffle(phi)
    return phi

#%%
# Generate MDP and markov(?) abstraction
mdp1 = MDP.generate(n_states=5, n_actions=2, sparsity=0, gamma=0.9)
phi = random_phi(mdp1.n_states-1)

mdp1.T
np.round(mdp1.R,2)

random_weights = random_transition_matrix((1,2))

agg_states = ((phi.sum(axis=0)>1) @ phi.transpose()).astype(bool)
other_states = ((phi.sum(axis=0)==1) @ phi.transpose()).astype(bool)
R = np.copy(mdp1.R)
T = np.copy(mdp1.T)
for a in range(mdp1.n_actions):
    R[a][agg_states[:,None]*agg_states] = np.mean(mdp1.R[a][agg_states[:,None]*agg_states])
    R[a][other_states[:,None]*agg_states] = np.mean(mdp1.R[a][other_states[:,None]*agg_states])
    R[a][agg_states[:,None]*other_states] = np.mean(mdp1.R[a][agg_states[:,None]*other_states])
    # R[a][other_states[:,None]*other_states] = mdp1.R[a][other_states[:,None]*other_states]
    T[a][:,agg_states] = random_weights * np.sum(mdp1.T[a][:,agg_states],axis=1, keepdims=True)
    # T[a][:,other_states] = random_transition_matrix((1,mdp1.n_states-2)) * np.sum(mdp1.T[a][:,other_states],axis=1, keepdims=True)
    assert(is_stochastic(T[a]))
mdp1.R = R
mdp1.T = T
print(random_weights)
print(phi)
print(T)
print(R)
#%%

v_star, q_star, pi_star = vi(mdp1)
v_star, pi_star

mdp2 = AbstractMDP(mdp1, phi)
assert is_markov(mdp2)
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
    pr_x = mdp1.stationary_distribution(pi=pi)
    belief = mdp2.B(pr_x)
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

    pr_x = mdp1.stationary_distribution(pi=pi,max_steps=400)
    belief = mdp2.B(pr_x)
    v_phi = belief @ v

    pi_list.append(pi)
    v_list.append(v)
    v_phi_list.append(v_phi)

#%%
for i in range(n_policies):
    pi_i = pi_list[i]
    v_i = v_list[i]
    v_phi_i = v_phi_list[i]
    for j in range(n_policies):
        if j==i:
            continue
        pi_j = pi_list[j]
        v_j = v_list[j]
        v_phi_j = v_phi_list[j]

        for z in range(mdp2.n_states):
            z_eq_mask = ~np.isclose(v_phi_i,v_phi_j, atol=1e-4)
            x_eq_mask = ~np.isclose(v_i,v_j,atol=1e-4)
            if not np.all(((x_eq_mask @ phi)>0) == z_eq_mask):
                print('incompatible masks??')
                break

            v_phi_gt = v_phi_i[z_eq_mask] > v_phi_j[z_eq_mask]
            v_gt = v_i[x_eq_mask] > v_j[x_eq_mask]

            v_phi_lt = v_phi_i[z_eq_mask] < v_phi_j[z_eq_mask]
            v_lt = v_i[x_eq_mask] < v_j[x_eq_mask]
            if (np.all(v_gt) and not np.all(v_phi_gt)) or (np.all(v_lt) and not np.all(v_phi_lt)):
                print('hypothesis might be wrong!')
                print(pi_star)
                print(pi_phi_star @ phi.transpose().astype(int))
                break
        else:
            continue
        break
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
