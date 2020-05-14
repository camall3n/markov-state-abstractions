import numpy as np
from mdpgen.mdp import MDP, AbstractMDP, UniformAbstractMDP, random_sparse_mask, random_transition_matrix, is_stochastic

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
    E_ratio_gnd = mdp_abs.B(pi_gnd) @ ratio_gnd
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

def has_block_dynamics(abstract_mdp):
    # Check that T(x'|a,x)/T(z'|a,x) is constant for all a and all x in z
    mdp_abs = abstract_mdp
    mdp_gnd = mdp_abs.base_mdp
    phi = mdp_abs.phi
    T_xax = mdp_gnd.T # T(x'|a,x)
    T_zax = mdp_gnd.T @ phi @ phi.transpose() # T(z'|a,x) in shape of T(x'|a,x)
    block_dynamics_ratio = np.divide(mdp_gnd.T, T_zax,
        out=np.zeros_like(mdp_gnd.T), where=T_zax!=0)
    # average along a,x dimensions where ratio is non-zero
    mean_ratio = np.average(block_dynamics_ratio, axis=(0,1),
                            weights=block_dynamics_ratio>0)
    # block dynamics ratio should be constant for all a,x (unless it's undefined)
    if not np.all(np.isclose(block_dynamics_ratio, mean_ratio)[block_dynamics_ratio>0]):
        return False
    return True

def is_hutter_markov(abstract_mdp):
    # Check that T(z'|a,x) = T(z'|a,z) for all x in z
    mdp_abs = abstract_mdp
    mdp_gnd = mdp_abs.base_mdp
    phi = mdp_abs.phi
    T_zax = mdp_gnd.T @ phi # T(z'|a,x)
    T_zaz = phi @ mdp_abs.T # T(z'|a,z) in shape of T(z'|a,x)
    if not np.allclose(T_zax, T_zaz):
        return False
    return True

def is_gdk_markov(abstract_mdp):
    # Check that T(x'|a,x) = T(x'|a,z) for all x in z
    raise NotImplementedError
    # Need to define some sort of weighting scheme, otherwise T(x'|a,z) is
    # not defined. It's supposed to be \sum_x T(x'|a,x)w(x|z).

def random_phi(n_states, n_abs_states):
    phi = np.eye(n_abs_states)
    mask = random_sparse_mask((1,n_abs_states), sparsity=1)
    mask = np.stack([mask.squeeze()]*(n_states-n_abs_states))
    phi = np.concatenate((phi, mask))
    np.random.shuffle(phi)
    return phi

#%%
def generate_non_markov_mdp_pair(n_states, n_abs_states, n_actions, sparsity=0, gamma=0.9, fixed_w=False):
    while True:
        mdp_gnd = MDP.generate(n_states=n_states, n_actions=n_actions, sparsity=sparsity, gamma=gamma)
        assert n_abs_states < n_states
        phi = random_phi(n_states, n_abs_states)
        if fixed_w:
            mdp_abs = UniformAbstractMDP(mdp_gnd, phi)
        else:
            mdp_abs = AbstractMDP(mdp_gnd, phi)

        # Ensure non-markov by checking inverse models and ratios
        if not is_markov(mdp_abs):
            break
    return mdp_gnd, mdp_abs

def generate_markov_mdp_pair(n_states, n_abs_states, n_actions, sparsity=0, gamma=0.9,
        equal_block_rewards=True, equal_block_transitions=True):
    # Sometimes numerical precision causes the abstract mdp to appear non-Markov
    # so we just keep generating until the problem goes away. Usually it's fine.
    while True:
        # generate an MDP and an abstraction function
        mdp_gnd = MDP.generate(n_states=n_states, n_actions=n_actions, sparsity=sparsity, gamma=gamma)
        assert n_abs_states < n_states
        phi = random_phi(n_states, n_abs_states)

        agg_states = ((phi.sum(axis=0)>1) @ phi.transpose()).astype(bool)
        other_states = ((phi.sum(axis=0)==1) @ phi.transpose()).astype(bool)

        random_weights = random_transition_matrix((1,n_states-n_abs_states+1))

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

        p0 = random_transition_matrix((1,n_states)).squeeze()
        mdp_abs = AbstractMDP(mdp_gnd, phi, p0=p0)

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
