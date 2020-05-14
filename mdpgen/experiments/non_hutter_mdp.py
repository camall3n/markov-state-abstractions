import gmpy
import numpy as np

from mdpgen.mdp import MDP, AbstractMDP, one_hot
from mdpgen.vi import vi

#%%
T0 = np.array([
    [0,   3/4, 1/4],
    [1/2, 3/8, 1/8],
    [1/2, 3/8, 1/8],
])
T1 = np.array([
    [0, 3/4, 1/4],
    [1/3, 1/2, 1/6],
    [1, 0, 0],
])
T2 = np.array([
    [0, 3/4, 1/4],
    [2/3, 1/4, 1/12],
    [0, 3/4, 1/4],
])
# T_alt = np.array([
#     [1/2, 3/8, 1/8],
#     [1, 0, 0],
#     [1, 0, 0],
# ])
R = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0],
])
mdp1 = MDP([T1, T2], [R, R], gamma=0.9)
v_star, q_star, pi_star = vi(mdp1)
v_star, pi_star

phi = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
])

mdp2 = AbstractMDP(mdp1, phi)
v_phi_star, q_phi_star, pi_phi_star = vi(mdp2)
v_phi_star

n_policies = mdp1.n_actions**mdp1.n_states
def get_policy(mdp, i):
    assert i < n_policies
    pi_string = gmpy.digits(i, mdp.n_actions).zfill(mdp.n_states)
    pi = np.asarray(list(pi_string), dtype=int)
    return pi

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
            z_mask = one_hot(z, mdp2.n_states).astype(bool)
            x_mask = (z_mask @ phi.transpose()).astype(bool)

            v_phi_eq = np.isclose(v_phi_i[z_mask],v_phi_j[z_mask])
            v_eq = np.isclose(v_i[x_mask],v_j[x_mask])
            if np.all(v_phi_eq):
                continue

            v_phi_gt = v_phi_i[z_mask] > v_phi_j[z_mask]
            v_gt = v_i[x_mask] > v_j[x_mask]
            v_phi_lt = v_phi_i[z_mask] < v_phi_j[z_mask]
            v_lt = v_i[x_mask] < v_j[x_mask]
            if not (np.all(v_gt == v_phi_gt) and np.all(v_lt == v_phi_lt)):
                print(i, pi_i, v_i, v_phi_i)
                print(j, pi_j, v_j, v_phi_j)
                print()
                break
        else:
            continue
        break
    else:
        continue
    break

    # compare V^pi vs V_phi^pi
    # print(i, pi, v_pi, v_phi_pi)

# (mdp.T[0] * mdp.R[0]).sum(axis=1)
# (mdp.T[1] * mdp.R[1]).sum(axis=1)

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
            v_phi_eq = np.isclose(v_phi_i,v_phi_j)
            v_eq = np.isclose(v_i,v_j)
            if np.any(v_phi_eq) or np.any(v_eq):
                continue

            v_phi_gt = v_phi_i > v_phi_j
            v_gt = v_i > v_j

            v_phi_lt = v_phi_i < v_phi_j
            v_lt = v_i < v_j
            if (np.all(v_gt) and not np.all(v_phi_gt)) or (np.all(v_lt) and not np.all(v_phi_lt)):
                print(i, pi_i)
                print(v_i)
                print(v_phi_i)
                print()
                print(j, pi_j)
                print(v_j)
                print(v_phi_j)
                print()
                break
        else:
            continue
        break
    else:
        continue
    break

    # compare V^pi vs V_phi^pi
    # print(i, pi, v_pi, v_phi_pi)

# (mdp.T[0] * mdp.R[0]).sum(axis=1)
# (mdp.T[1] * mdp.R[1]).sum(axis=1)
