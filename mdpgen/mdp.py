import copy
import numpy as np

def normalize(M):
    denoms = M.sum(axis=1)
    M = M / denoms[:,None]
    return M

def is_stochastic(M):
    return np.allclose(M, normalize(M))

def random_sparse_mask(size, sparsity):
    n_rows, n_cols = size
    p = (1-sparsity)# probability of 1
    q = (n_cols*p - 1)/(n_cols-1)# get remaining probability after mandatory 1s
    if 0 < q <= 1:
        some_ones = np.random.choice([0,1],size=(n_rows, n_cols-1),p=[1-q, q])
        mask = np.concatenate([np.ones((n_rows,1)), some_ones], axis=1)
    else:
        mask = np.concatenate([np.ones((n_rows,1)), np.zeros((n_rows, n_cols-1))], axis=1)
    for row in mask:
        np.random.shuffle(row)
    return mask

def random_transition_matrix(size):
    T = normalize(np.random.rand(*size))
    return T

def random_reward_matrix(Rmin, Rmax, size):
    R = np.random.uniform(Rmin, Rmax, size)
    R = np.round(R, 2)
    return R

def random_observation_fn(n_states, n_obs_per_block):
    observation_fns = []
    for s in range(n_states):
        Ob_s = random_transition_matrix(size=(1,n_obs_per_block))
        observation_fns.append(Ob_s)
    obs_fn_stack = np.stack(observation_fns).squeeze()
    obs_fn_mask = np.kron(np.eye(n_states),np.ones((1,n_obs_per_block)))
    obs_fn_tiled = np.kron(np.ones((1,n_states)),obs_fn_stack)
    observation_fn = obs_fn_mask*obs_fn_tiled
    return observation_fn

def one_hot(x, n):
    return np.eye(n)[x]

def condition_T_on_pi(T, pi):
    n_actions = len(T)
    n_states = len(T[0])
    pi_one_hot = one_hot(pi, n_actions)
    T_stack = np.stack(T, axis=0)
    T_pi = np.tensordot(pi_one_hot, T, axes=1)[pi, np.arange(n_states), :]
    return T_pi

class MDP:
    def __init__(self, T, R, gamma=0.9):
        self.n_states = len(T[0])
        self.n_actions = len(T)
        self.gamma = gamma
        self.T = copy.deepcopy(T)
        self.R = copy.deepcopy(R)
        self.R_min = np.min(np.stack(self.R))
        self.R_max = np.max(np.stack(self.R))

    @classmethod
    def generate(cls, n_states, n_actions, sparsity=0, gamma=0.9, Rmin=-1, Rmax=1):
        T = []# List of s -> s transition matrices, one for each action
        R = []# List of s -> s reward matrices, one for each action
        for a in range(n_actions):
            T_a = random_transition_matrix(size=(n_states, n_states))
            R_a = random_reward_matrix(Rmin, Rmax, (n_states, n_states))
            if sparsity > 0:
                mask = random_sparse_mask((n_states, n_states), sparsity)
                T_a = normalize(T_a*mask)
                R_a = R_a*mask
            T.append(T_a)
            R.append(R_a)
        mdp = cls(T, R, gamma)
        return mdp

class BlockMDP(MDP):
    def __init__(self, base_mdp, n_obs_per_block=2, obs_fn=None):
        super().__init__(base_mdp.T, base_mdp.R, base_mdp.gamma)
        self.base_mdp = copy.deepcopy(base_mdp)
        self._n_obs_per_block = n_obs_per_block
        self.n_states = base_mdp.n_states*n_obs_per_block

        if obs_fn is None:
            obs_fn = random_observation_fn(base_mdp.n_states, n_obs_per_block)
        else:
            n_obs_per_block = obs_fn.shape[1]

        obs_mask = (obs_fn > 0).astype(int)

        self.T = []# List of x -> x transition matrices, one for each action
        self.R = []# List of x -> x reward matrices, one for each action
        for a in range(self.n_actions):
            Ta, Ra = base_mdp.T[a], base_mdp.R[a]
            Tx_a = obs_mask.transpose() @ Ta @ obs_fn
            # Tx_a = np.kron((Ta @ obs_fn), np.ones((n_obs_per_block,1)))
            # Tx_a = np.kron(Ta, np.ones((n_obs_per_block,n_obs_per_block)))
            # Tx_a = (obs_fn.transpose() @ Ta) @ obs_fn
            Rx_a = obs_mask.transpose() @ Ra @ obs_mask
            # Rx_a = np.kron(Ra, np.ones((n_obs_per_block,n_obs_per_block)))
            self.T.append(Tx_a)
            self.R.append(Rx_a)
        self.obs_fn = obs_fn

class AbstractMDP(MDP):
    def __init__(self, base_mdp, phi, pi=None, p0=None):
        super().__init__(base_mdp.T, base_mdp.R, base_mdp.gamma)
        self.base_mdp = copy.deepcopy(base_mdp)
        self.phi = phi# array: base_mdp.n_states, n_abstract_states
        self.n_states = phi.shape[-1]
        self.n_obs = base_mdp.n_states

        # compute steady-state distribution in base mdp
        state_distr = self.compute_steady_state_distr(p0)

        belief = self.B(state_distr)
        self.T = [self.compute_Tz(belief,T_a)
                    for T_a in base_mdp.T]
        self.R = [self.compute_Rz(belief,Rx_a,Tx_a,Tz_a)
                    for (Rx_a, Tx_a, Tz_a) in zip(base_mdp.R, base_mdp.T, self.T)]
        self.Rmin = np.min(np.stack(self.R))
        self.Rmax = np.max(np.stack(self.R))

    def compute_steady_state_distr(self, pi=None, p0=None, n_steps=200):
        if p0 is None:
            state_distr = np.ones(self.base_mdp.n_states)/self.base_mdp.n_states
        else:
            state_distr = p0
        old_distr = state_distr
        for t in range(n_steps):
            if pi is None:
                # Assume uniform random policy
                T = np.mean(np.stack(self.base_mdp.T,axis=0),axis=0)
                R = np.mean(np.stack(self.base_mdp.R,axis=0),axis=0)
                # pi_t = np.random.randint(0, self.n_actions, size=(self.base_mdp.n_states,))
                # T = condition_T_on_pi(self.base_mdp.T, pi_t)
                # R = condition_T_on_pi(self.base_mdp.R, pi_t)
            else:
                pi_t = pi
                T = condition_T_on_pi(self.base_mdp.T, pi_t)
                R = condition_T_on_pi(self.base_mdp.R, pi_t)
            state_distr = T @ state_distr
            if np.allclose(state_distr, old_distr):
                break
            old_distr = state_distr
        return state_distr

    def B(self, p):
        return normalize(p*self.phi.transpose())

    def compute_Tz(self, belief, Tx):
        return belief @ Tx @ self.phi

    def compute_Rz(self, belief, Rx, Tx, Tz):
        return np.divide( (belief@(Rx*Tx)@self.phi), Tz,
                         out=np.zeros_like(Tz), where=(Tz!=0) )


# p = random_transition_matrix((1,10))
# phi = random_sparse_mask((10,3),0.9)
# p.shape
# phi.shape
# pz = p @ phi
# pz.shape
# B = normalize(p*phi.transpose())
# B.shape
# T = random_transition_matrix((10,10))
# np.round(B,2)
# np.round(T,2)
#
# (B @ T @ phi).shape


mdp1 = MDP.generate(n_states=3, n_actions=2, sparsity=0.5)
mdp2 = BlockMDP(mdp1, n_obs_per_block=2)
phi = np.asarray(mdp2.obs_fn.transpose()>0,dtype=int)
mdp3 = AbstractMDP(mdp2, phi)
assert np.allclose(mdp1.T, mdp3.T)
assert np.allclose(mdp1.R, mdp3.R)

# np.ceil(mdp.T[0])
# mdp.R
# mdp.base_mdp().T[a]
# np.kron( (mdp.base_mdp().T[a] @ mdp.Ob), np.ones((2,1)))
#
# mdp.T[a] @ mdp.Ob
#
# assert all([is_stochastic(mdp.T[a]) for a in range(mdp.n_actions)])
