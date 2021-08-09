import gmpy
import numpy as np

from mdpgen.mdp import MDP, AbstractMDP
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

# for each ground-state policy
n_policies = mdp1.n_actions**mdp1.n_states
for i in range(n_policies):
    pi_string = gmpy.digits(i, mdp1.n_actions).zfill(mdp1.n_states)
    pi = np.asarray(list(pi_string), dtype=int)

    # compare V^pi vs V_phi^pi
    v_pi = vi(mdp1, pi)[0]
    belief = mdp2.B(pi)
    v_phi_pi = belief @ v_pi
    print(i, pi, v_pi, v_phi_pi)

# (mdp.T[0] * mdp.R[0]).sum(axis=1)
# (mdp.T[1] * mdp.R[1]).sum(axis=1)
