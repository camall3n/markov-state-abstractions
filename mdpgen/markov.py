import numpy as np
from mdpgen.mdp import MDP, AbstractMDP, random_sparse_mask, random_transition_matrix, is_stochastic

def matching_I(mdp_gnd, mdp_abs, pi_gnd, pi_abs):
    # ground mdp, abstract mdp, ground policy, abstract policy
    # Does I(z',z) = I(x',x)?
    phi = mdp_abs.phi
    Ig = mdp_gnd.get_I(pi_gnd)
    Ia = mdp_abs.get_I(pi_abs)
    Ia_grounded = np.zeros_like(Ig)
    for a in range(mdp_gnd.n_actions):
        Ia_grounded[a,:,:] = phi @ Ia[a,:,:] @ phi.transpose()
    Ia_grounded *= (mdp_gnd.get_N(pi_gnd) > 0)
    if not np.all(Ig == Ia_grounded):
        return False
    return True

def matching_ratios(mdp_gnd, mdp_abs, pi_gnd, pi_abs):
    # ground mdp, abstract mdp, ground policy, abstract policy
    # does N_phi(z'|z) / P_z' = E_B(x|z)[ N(x'|x) / P_x'] ?
    phi = mdp_abs.phi
    N_gnd = mdp_gnd.get_N(pi_gnd)
    Px = mdp_gnd.stationary_distribution(pi_gnd)
    N_abs = mdp_abs.get_N(pi_abs)
    Pz = mdp_abs.stationary_distribution(pi_abs)

    ratio_abs = np.divide(N_abs, Pz[None,:], out=np.zeros_like(N_abs), where=Pz!=0)
    ratio_gnd = np.divide(N_gnd, Px[None,:], out=np.zeros_like(N_gnd), where=Px!=0)
    E_ratio_gnd = mdp_abs.B(Px) @ ratio_gnd
    if not np.allclose(E_ratio_gnd, ratio_abs @ phi.transpose(), atol=1e-6):
        return False
    return True

def is_markov(abstract_mdp):
    mdp_abs = abstract_mdp
    mdp_gnd = mdp_abs.base_mdp
    phi = mdp_abs.phi
    for pi_gnd in mdp_abs.piecewise_constant_policies():
        pi_abs = mdp_abs.get_abstract_policy(pi_gnd)
        if not matching_I(mdp_gnd, mdp_abs, pi_gnd, pi_abs):
            return False
        if not matching_ratios(mdp_gnd, mdp_abs, pi_gnd, pi_abs):
            return False
    return True

def random_phi(n_abs_states):
    phi = np.eye(n_abs_states)
    phi = np.concatenate((phi, random_sparse_mask((1,n_abs_states), sparsity=1)))
    np.random.shuffle(phi)
    return phi

#%%
def generate_non_markov_mdp_pair(n_states, n_abs_states, n_actions):
    while True:
        mdp_gnd = MDP.generate(n_states=n_states, n_actions=n_actions, sparsity=0, gamma=0.9)
        assert n_abs_states < n_states
        phi = random_phi(n_abs_states)
        mdp_abs = AbstractMDP(mdp_gnd, phi)

        # Ensure non-markov by checking inverse models and ratios
        if not is_markov(mdp_abs):
            break
    return mdp_gnd, mdp_abs

def generate_markov_mdp_pair(n_states, n_abs_states, n_actions, equal_block_rewards=True, equal_block_transitions=True):
    # Sometimes numerical precision causes the abstract mdp to appear non-Markov
    # so we just keep generating until the problem goes away. Usually it's fine.
    while True:
        # generate an MDP and an abstraction function
        mdp_gnd = MDP.generate(n_states=n_states, n_actions=n_actions, sparsity=0, gamma=0.9)
        assert n_abs_states < n_states
        phi = random_phi(n_abs_states)


        random_weights = random_transition_matrix((1,2))
        random_weights = np.ones((1,2))/2

        agg_states = ((phi.sum(axis=0)>1) @ phi.transpose()).astype(bool)
        other_states = ((phi.sum(axis=0)==1) @ phi.transpose()).astype(bool)

        # adjust T and R to achieve desired properties
        R = np.copy(mdp_gnd.R)
        T = np.copy(mdp_gnd.T)
        for a in range(mdp_gnd.n_actions):
            if equal_block_rewards:
                R[a][agg_states[:,None]*agg_states] = np.mean(mdp_gnd.R[a][agg_states[:,None]*agg_states])
                R[a][other_states[:,None]*agg_states] = np.mean(mdp_gnd.R[a][other_states[:,None]*agg_states])
                R[a][agg_states[:,None]*other_states] = np.mean(mdp_gnd.R[a][agg_states[:,None]*other_states])

            T[a][:,agg_states] = random_weights * np.sum(mdp_gnd.T[a][:,agg_states],axis=1, keepdims=True)
            if equal_block_transitions:
                T[a][agg_states] = np.mean(T[a][agg_states,:],axis=0)
                T[a][agg_states][:,agg_states] = random_weights * np.sum(T[a][agg_states][:,agg_states],axis=1, keepdims=True)
            # T[a][:,other_states] = random_transition_matrix((1,mdp_gnd.n_states-2)) * np.sum(mdp_gnd.T[a][:,other_states],axis=1, keepdims=True)
            assert(is_stochastic(T[a]))
        mdp_gnd.R = R
        mdp_gnd.T = T

        mdp_abs = AbstractMDP(mdp_gnd, phi)

        # Ensure that the abstraction is markov by checking inverse models and ratios
        if is_markov(mdp_abs):
            break
    return mdp_gnd, mdp_abs

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
    mdp_gnd = MDP([T1, T2], [R, R], gamma=0.9)
    phi = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    mdp_abs = AbstractMDP(mdp_gnd, phi)
    assert is_markov(mdp_abs)

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
    mdp_gnd = MDP([T, T], [R, R], gamma=0.9)
    phi = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    mdp_abs = AbstractMDP(mdp_gnd, phi)
    # Even though this abstract MDP is Markov, is_markov() will return False,
    # since its conditions (while sufficient) are stricter than necessary
    assert not is_markov(mdp_abs)

if __name__ == '__main__':
    test()
    test_non_markov_B()
    print('All tests passed.')
