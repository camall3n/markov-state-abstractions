import numpy as np

from mdpgen.blockmdp import MDP, BlockMDP
from mdpgen.vi import vi

mdp = BlockMDP(n_blocks=5, n_actions=6, n_obs_per_block=2)
v, q, pi = vi(mdp)
v
pi


v_alt = np.zeros_like(v)
for s in range(mdp.n_states):
    v_alt[s] = q[pi[s]][s]
v_alt = v_alt.squeeze()
assert np.allclose(v_alt, v)

v_pi = vi(mdp, pi)[0]
assert np.allclose(v_pi, v)

m_phi = mdp.get_abstract_mdp()
v_phi, q_phi, pi_phi = vi(m_phi)
v_phi
pi_phi

#%%
mdp = MDP(n_states=3, n_actions=2, gamma=0.9)
mdp.T[0] = np.array([
    [0, 3/4, 1/4],
    [1/3, 1/2, 1/6],
    [1, 0, 0],
])
mdp.T[1] = np.array([
    [0, 3/4, 1/4],
    [2/3, 1/4, 1/12],
    [0, 3/4, 1/4],
])
mdp.R[0] = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
])
mdp.R[1] = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 0]
])

v_star, q_star, pi_star = vi(mdp)
v_star
pi_star

for a0 in [0,1]:
    for a1 in [0,1]:
        for a2 in [0,1]:
            pi_phi = [a0, a1, a2]
            v_pi_phi = vi(mdp, pi_phi)[0]
            print(pi_phi, v_pi_phi, np.round(np.dot(np.array([0, 3/4, 1/4]),v_pi_phi),8))
