import gmpy
import numpy as np
from mdpgen.mdp import MDP, AbstractMDP, normalize, one_hot

def get_policy(mdp, i):
    assert i < mdp.n_actions**mdp.n_states
    pi_string = gmpy.digits(i, mdp.n_actions).zfill(mdp.n_states)
    pi = np.asarray(list(pi_string), dtype=int)
    return pi

def all_policies(mdp):
    policies = []
    n_policies = mdp.n_actions**mdp.n_states
    for i in range(n_policies):
        pi = get_policy(mdp, i)
        policies.append(pi)
    return policies

def is_abstract_policy(pi, phi):
    agg_states = ((phi.sum(axis=0)>1) @ phi.transpose()).astype(bool)
    return np.all(pi[agg_states] == pi[agg_states][0])

def get_abstract_policy(pi, phi):
    assert is_abstract_policy(pi, phi)
    mask = phi.transpose()
    obs_fn = normalize(mask)
    return (pi @ obs_fn.transpose()).astype(int)

def abstract_policies(mdp, phi):
    return [pi for pi in all_policies(mdp) if is_abstract_policy(pi, phi)]

def get_N(mdp, pi):
    return mdp.T_pi(pi)

def get_I(mdp, pi):
    pi_one_hot = one_hot(pi,mdp.n_actions).transpose()[:,:,None]
    mdp.T
    N = get_N(mdp, pi)[None,:,:]
    I = np.divide(mdp.T*pi_one_hot, N, out=np.zeros_like(mdp.T), where=N!=0)
    # mdp.T.shape
    # N.shape
    # pi_one_hot.shape
    # np.argmax(I,0)
    return I

def matching_I(m_g, m_a, pi_g, pi_a):
    phi = m_a.phi
    Ig = get_I(m_g,pi_g)
    Ia = get_I(m_a,pi_a)
    Ia_grounded = np.zeros_like(Ig)
    for a in range(m_g.n_actions):
        Ia_grounded[a,:,:] = phi @ Ia[a,:,:] @ phi.transpose()
    Ia_grounded *= (get_N(m_g, pi_g) > 0)
    if not np.all(Ig == Ia_grounded):
        return False
    return True

def matching_ratios(m_g, m_a, pi_g, pi_a):
    phi = m_a.phi
    Ng = get_N(m_g,pi_g)
    Px = m_g.stationary_distribution(pi_g)
    Na = get_N(m_a,pi_a)
    Pz = m_a.stationary_distribution(pi_a)

    ratio_a = np.divide(Na, Pz[None,:], out=np.zeros_like(Na), where=Pz!=0)
    ratio_g = np.divide(Ng, Px[None,:], out=np.zeros_like(Ng), where=Px!=0)
    E_ratio_g = m_a.B(Px) @ ratio_g
    if not np.allclose(E_ratio_g, ratio_a @ phi.transpose()):
        return False
    return True

def is_markov(abstract_mdp):
    m_a = abstract_mdp
    m_g = m_a.base_mdp
    phi = m_a.phi
    for pi_g in abstract_policies(m_g, phi):
        pi_a = get_abstract_policy(pi_g, phi)
        if not matching_I(m_g, m_a, pi_g, pi_a):
            return False
        if not matching_ratios(m_g, m_a, pi_g, pi_a):
            return False
    return True

#%%
def test():
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
    R = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ])
    m_g = MDP([T1, T2], [R, R], gamma=0.9)
    phi = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    m_a = AbstractMDP(m_g, phi)
    assert is_markov(m_a)
    print('All tests passed.')

if __name__ == '__main__':
    test()
