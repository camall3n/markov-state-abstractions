import copy
import numpy as np

def is_stochastic(M):
    N = M / M.sum(axis=1)[:,None]
    return np.allclose(M,N)

def random_stochastic_matrix(*size):
    M = np.random.rand(*size)
    M = M / M.sum(axis=1)[:,None]
    return M

def random_reward_matrix(*size):
    R = random_stochastic_matrix(*size)
    R = np.round(R*len(R)*10-5, 2)
    return R

def random_observation_fn(n_states, n_obs_per_block):
    observation_fns = []
    for s in range(n_states):
        Ob_s = random_stochastic_matrix(1,n_obs_per_block)
        observation_fns.append(Ob_s)
    obs_fn_stack = np.stack(observation_fns).squeeze()
    obs_fn_mask = np.kron(np.eye(n_states),np.ones((1,n_obs_per_block)))
    obs_fn_tiled = np.kron(np.ones((1,n_states)),obs_fn_stack)
    observation_fn = obs_fn_mask*obs_fn_tiled
    return observation_fn

class MDP:
    def __init__(self, n_states, n_actions, gamma=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        self.T = []# List of s -> s transition matrices, one for each action
        self.R = []# List of s -> s reward matrices, one for each action
        for a in range(n_actions):
            T_a = random_stochastic_matrix(n_states, n_states)
            R_a = random_reward_matrix(n_states, n_states)
            self.T.append(T_a)
            self.R.append(R_a)
        self.R_min = np.min(np.stack(self.R))
        self.R_max = np.max(np.stack(self.R))


class BlockMDP(MDP):
    def __init__(self, n_blocks, n_actions, n_obs_per_block, gamma=0.9):
        super().__init__(n_states=n_blocks, n_actions=n_actions, gamma=gamma)
        self._true_mdp = copy.deepcopy(self)
        self._n_blocks = n_blocks
        self._n_obs_per_block = n_obs_per_block
        self.n_states = n_blocks*n_obs_per_block
        # self.n_actions = n_actions
        # self.gamma = gamma
        # self.R_min =

        _Ob_fn = random_observation_fn(n_blocks, n_obs_per_block)
        self.T = []# List of x -> x transition matrices, one for each action
        self.R = []# List of x -> x reward matrices, one for each action
        for a in range(n_actions):
            Ta, Ra = self._true_mdp.T[a], self._true_mdp.R[a]
            Tx_a = np.kron(np.matmul(Ta,_Ob_fn),np.ones((2,1)))
            # Tx_a = np.kron(Ta, np.ones((n_obs_per_block,n_obs_per_block)))
            # Tx_a = np.matmul(np.matmul(_Ob_fn.transpose(), Ta), _Ob_fn)
            Rx_a = np.kron(Ra, np.ones((n_obs_per_block,n_obs_per_block)))
            self.T.append(Tx_a)
            self.R.append(Rx_a)
        self.Ob = _Ob_fn

    def get_abstract_mdp(self):
        return self._true_mdp


# mdp = BlockMDP(n_blocks=3, n_actions=2, n_obs_per_block=2)
# mdp.Ob
# mdp.get_abstract_mdp().T[a]
# np.kron(np.matmul(mdp.get_abstract_mdp().T[a],mdp.Ob),np.ones((2,1)))
#
# np.matmul(mdp.T[a],mdp.Ob)
#
# assert all([is_stochastic(mdp.T[a]) for a in range(mdp.n_actions)])
