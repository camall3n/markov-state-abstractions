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
    def __init__(self, base_mdp, n_obs_per_block):
        super().__init__(base_mdp.T, base_mdp.R, base_mdp.gamma)
        self.base_mdp = copy.deepcopy(base_mdp)
        self._n_blocks = base_mdp.n_states
        self._n_obs_per_block = n_obs_per_block
        self.n_states = base_mdp.n_states*n_obs_per_block

        _Ob_fn = random_observation_fn(base_mdp.n_states, n_obs_per_block)
        self.T = []# List of x -> x transition matrices, one for each action
        self.R = []# List of x -> x reward matrices, one for each action
        for a in range(self.n_actions):
            Ta, Ra = base_mdp.T[a], base_mdp.R[a]
            Tx_a = np.kron(np.matmul(Ta,_Ob_fn),np.ones((n_obs_per_block,1)))
            # Tx_a = np.kron(Ta, np.ones((n_obs_per_block,n_obs_per_block)))
            # Tx_a = np.matmul(np.matmul(_Ob_fn.transpose(), Ta), _Ob_fn)
            Rx_a = np.kron(Ra, np.ones((n_obs_per_block,n_obs_per_block)))
            self.T.append(Tx_a)
            self.R.append(Rx_a)
        self.Ob = _Ob_fn

# mdp = MDP.generate(n_states=3, n_actions=2, sparsity=0.5)
# mdp.T
# mdp.R
# mdp = BlockMDP(mdp, n_obs_per_block=2)
# np.ceil(mdp.T[0])
# mdp.R
# mdp.base_mdp().T[a]
# np.kron(np.matmul(mdp.base_mdp().T[a],mdp.Ob),np.ones((2,1)))
#
# np.matmul(mdp.T[a],mdp.Ob)
#
# assert all([is_stochastic(mdp.T[a]) for a in range(mdp.n_actions)])
