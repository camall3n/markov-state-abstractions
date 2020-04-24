import gmpy
import numpy as np

from mdpgen.mdp import MDP, BlockMDP, AbstractMDP, normalize
from mdpgen.vi import vi

def test_vi():
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
    pi_phi_grounded = np.kron(pi_phi, np.ones((1,mdp.n_states//m_phi.n_states)))
    assert np.allclose(pi_phi_grounded, pi)
    print('All tests passed.')
test_vi()

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

pr_x = mdp1.stationary_distribution()
phi = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
])

mdp2 = AbstractMDP(mdp1, phi, p0=pr_x)
v_phi_star, q_phi_star, pi_phi_star = vi(mdp2)
v_phi_star

# for each ground-state policy
for i in range(mdp1.n_actions**mdp1.n_states):
    pi_string = gmpy.digits(i, mdp1.n_actions).zfill(mdp1.n_states)
    pi = np.asarray(list(pi_string), dtype=int)

    v_pi = vi(mdp1, pi)[0]
    v_phi_pi = mdp2.belief @ v_pi
    print(pi, v_pi, v_phi_pi)

# (mdp.T[0] * mdp.R[0]).sum(axis=1)
# (mdp.T[1] * mdp.R[1]).sum(axis=1)
