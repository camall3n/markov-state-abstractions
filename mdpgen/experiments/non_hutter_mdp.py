import gmpy
import numpy as np

from mdpgen.mdp import MDP, AbstractMDP, one_hot
from mdpgen.vi import vi
from mdpgen.markov import is_markov, has_block_dynamics, is_hutter_markov

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
mdp2 = AbstractMDP(MDP([T0, T1], [R, R], gamma=0.9), np.array([[1,0],[0,1],[0,1]]))
is_hutter_markov(mdp2)
is_markov(mdp2)
v_star, q_star, pi_star = vi(mdp1)
v_star, pi_star

phi = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
])

mdp2 = AbstractMDP(mdp1, phi)
assert is_markov(mdp2)
assert has_block_dynamics(mdp2)
assert not is_hutter_markov(mdp2)
