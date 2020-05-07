import numpy as np
from mdpgen.mdp import MDP, AbstractMDP, random_sparse_mask, random_transition_matrix, is_stochastic

def matching_I(m_g, m_a, pi_g, pi_a):
    # ground mdp, abstract mdp, ground policy, abstract policy
    # Does I(z',z) = I(x',x)?
    phi = m_a.phi
    Ig = m_g.get_I(pi_g)
    Ia = m_a.get_I(pi_a)
    Ia_grounded = np.zeros_like(Ig)
    for a in range(m_g.n_actions):
        Ia_grounded[a,:,:] = phi @ Ia[a,:,:] @ phi.transpose()
    Ia_grounded *= (m_g.get_N(pi_g) > 0)
    if not np.all(Ig == Ia_grounded):
        return False
    return True

def matching_ratios(m_g, m_a, pi_g, pi_a):
    # ground mdp, abstract mdp, ground policy, abstract policy
    # does N_phi / P_z = E_B(x|z)[ N / P_x] ?
    phi = m_a.phi
    Ng = m_g.get_N(pi_g)
    Px = m_g.stationary_distribution(pi_g)
    Na = m_a.get_N(pi_a)
    Pz = m_a.stationary_distribution(pi_a)

    ratio_a = np.divide(Na, Pz[None,:], out=np.zeros_like(Na), where=Pz!=0)
    ratio_g = np.divide(Ng, Px[None,:], out=np.zeros_like(Ng), where=Px!=0)
    E_ratio_g = m_a.B(Px) @ ratio_g
    if not np.allclose(E_ratio_g, ratio_a @ phi.transpose(), atol=1e-6):
        return False
    return True

def is_markov(abstract_mdp):
    m_a = abstract_mdp
    m_g = m_a.base_mdp
    phi = m_a.phi
    for pi_g in m_a.piecewise_constant_policies():
        pi_a = m_a.get_abstract_policy(pi_g)
        if not matching_I(m_g, m_a, pi_g, pi_a):
            return 'I mismatch'
        if not matching_ratios(m_g, m_a, pi_g, pi_a):
            return 'R mismatch'
    return True

def random_phi(n_abs_states):
    phi = np.eye(n_abs_states)
    phi = np.concatenate((phi, random_sparse_mask((1,n_abs_states), sparsity=1)))
    np.random.shuffle(phi)
    return phi

#%%
def generate_markov_mdp_pair(n_states, n_abs_states, n_actions, equal_block_rewards=True, equal_block_transitions=True):
    # Sometimes numerical precision causes the abstract mdp to appear non-Markov
    # so we just keep generating until the problem goes away. Usually it's fine.
    while True:
        # generate an MDP and an abstraction function
        mdp1 = MDP.generate(n_states=n_states, n_actions=n_actions, sparsity=0, gamma=0.9)
        assert n_abs_states < n_states
        phi = random_phi(n_abs_states)


        random_weights = random_transition_matrix((1,2))
        random_weights = np.ones((1,2))/2

        agg_states = ((phi.sum(axis=0)>1) @ phi.transpose()).astype(bool)
        other_states = ((phi.sum(axis=0)==1) @ phi.transpose()).astype(bool)

        # adjust T and R to achieve desired properties
        R = np.copy(mdp1.R)
        T = np.copy(mdp1.T)
        for a in range(mdp1.n_actions):
            if equal_block_rewards:
                R[a][agg_states[:,None]*agg_states] = np.mean(mdp1.R[a][agg_states[:,None]*agg_states])
                R[a][other_states[:,None]*agg_states] = np.mean(mdp1.R[a][other_states[:,None]*agg_states])
                R[a][agg_states[:,None]*other_states] = np.mean(mdp1.R[a][agg_states[:,None]*other_states])

            T[a][:,agg_states] = random_weights * np.sum(mdp1.T[a][:,agg_states],axis=1, keepdims=True)
            if equal_block_transitions:
                T[a][agg_states] = np.mean(T[a][agg_states,:],axis=0)
                T[a][agg_states][:,agg_states] = random_weights * np.sum(T[a][agg_states][:,agg_states],axis=1, keepdims=True)
            # T[a][:,other_states] = random_transition_matrix((1,mdp1.n_states-2)) * np.sum(mdp1.T[a][:,other_states],axis=1, keepdims=True)
            assert(is_stochastic(T[a]))
        mdp1.R = R
        mdp1.T = T

        mdp2 = AbstractMDP(mdp1, phi)

        # Ensure that the abstraction is markov by checking inverse models and ratios
        if is_markov(mdp2):
            break
    return mdp1, mdp2



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

def test_non_markov_B():
    T = np.array([
        [0, .5, .5, 0, 0, 0],
        [0, 0, 0, .5, .5, 0],
        [0, 0, 0, 0, .5, .5],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ])
    R = (T > 0).astype(float)
    m_g = MDP([T, T], [R, R], gamma=0.9)
    phi = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    m_a = AbstractMDP(m_g, phi)
    # Even though this abstract MDP is Markov, is_markov() will return False,
    # since its conditions (while sufficient) are stricter than necessary
    assert not is_markov(m_a)

if __name__ == '__main__':
    test()
    test_non_markov_B()
    print('All tests passed.')
