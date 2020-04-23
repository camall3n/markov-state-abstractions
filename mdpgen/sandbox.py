import gmpy
import numpy as np

from mdpgen.mdp import MDP, BlockMDP
from mdpgen.vi import vi


mdp = BlockMDP(MDP.generate(n_states=5, n_actions=6), n_obs_per_block=3)
v, q, pi = vi(mdp)

v_alt = np.zeros_like(v)
for s in range(mdp.n_states):
    v_alt[s] = q[pi[s]][s]
v_alt = v_alt.squeeze()
assert np.allclose(v_alt, v)

v_pi = vi(mdp, pi)[0]
assert np.allclose(v_pi, v)

m_phi = mdp.base_mdp
v_phi, q_phi, pi_phi = vi(m_phi)
pi_phi_grounded = np.kron(pi_phi, np.ones((1,mdp._n_obs_per_block)))
assert np.allclose(pi_phi_grounded, pi)

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
mdp = MDP([T1, T2], [R, R], gamma=0.9)
v_star, q_star, pi_star = vi(mdp)
v_star, pi_star

3/4*6.76955067 + 1/4*5.89598745


# (mdp.T[0] * mdp.R[0]).sum(axis=1)
# (mdp.T[1] * mdp.R[1]).sum(axis=1)

#%%

pi_list = []
for i in range(mdp.n_actions**mdp.n_states):
    pi_string = gmpy.digits(i, mdp.n_actions).zfill(mdp.n_states)
    pi_list.append(np.asarray(list(pi_string), dtype=int))
#%%
T = np.array([
    [0, 1],
    [.5, .5],
])
R = np.array([
    [0, 1],
    [1, 0]
])
mdp_phi = MDP([T],[R],gamma=0.9)
v_phi_star, q_phi_star, pi_phi_star = vi(mdp_phi)
v_phi_star
for pi in pi_list:
    v_pi = vi(mdp, pi)[0]
    v_phi_pi = np.array([v_pi[0], np.dot(np.array([0, 3/4, 1/4]),v_pi)])
    print(pi, v_pi, v_phi_pi)

#%%
# T = T1
for i in range(10):
    T = np.matmul(T,T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)
