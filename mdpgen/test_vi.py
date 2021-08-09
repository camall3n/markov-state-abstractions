import numpy as np

from mdpgen.mdp import MDP, BlockMDP
from mdpgen.vi import vi

def main():
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

if __name__ == '__main__':
    main()
