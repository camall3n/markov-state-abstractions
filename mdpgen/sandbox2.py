import gmpy
import numpy as np

from mdpgen.mdp import MDP, AbstractMDP, condition_T_on_pi, one_hot, normalize
from mdpgen.vi import vi

#%%
# Define 8-state MDP with 4 effective states
T = np.zeros((8,8))
T[0,1:3] = .5
T[1:3,3:6] = np.array([
    [.5, .5, 0],
    [.5, 0, .5],
])
T[3,6:] = .5
T[4:6,-1] = 1
T[6:,0] = 1

R = np.array([1,2,2,3,3,3,4,4])[None,:]*(T > 0).astype(int)/4

m8 = MDP([T], [R], gamma=0.9)
v_star, q_star, pi_star = vi(m8)
v_star

# Compute distributions over ground states at each timestep

pr8_0 = np.eye(8)[0]
pr8_1 = m8.image(pr8_0)
pr8_1a = np.eye(8)[1]
pr8_2 = m8.image(pr8_1)
pr8_2a = m8.image(pr8_1a)
pr8_3 = m8.image(pr8_2)
pr8_3a = m8.image(pr8_2a)

#%%
# Aggregate 3 states into 1 abstract state to form 6-state abstract MDP
phi6 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
])

m6 = AbstractMDP(m8, phi6)
v_phi_star, q_phi_star, pi_phi_star = vi(m6)

# compute distributions over abstract states at each timestep
pr6_0 = pr8_0 @ phi6
pr6_1 = pr8_1 @ phi6
pr6_1a = pr8_1a @ phi6
pr6_2 = pr8_2 @ phi6
pr6_2a = pr8_2a @ phi6
pr6_3 = pr8_3 @ phi6

# note that the following distributions map to different abstract states
pr8_1 @ phi6
pr8_1a @ phi6

# belief depends on history and is therefore non-Markov
B6_2 = m6.B(pr8_2)
B6_2a = m6.B(pr8_2a)
assert not np.allclose(B6_2, B6_2a)

# however, T_phi does NOT depend on history, and IS Markov
T6_2 = B6_2 @ m6.base_mdp.T[0] @ phi6
T6_2a = B6_2a @ m6.base_mdp.T[0] @ phi6
assert np.allclose(T6_2, T6_2a)

#%%
# Aggregate more states to form a 4-state abstract MDP
phi4 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

m4 = AbstractMDP(m8, phi4)
v_phi_star, q_phi_star, pi_phi_star = vi(m4)

# compute distributions over abstract states at each timestep
pr4_0 = pr8_0 @ phi4
pr4_1 = m4.image(pr4_0)
pr4_2 = m4.image(pr4_1)
pr4_3 = m4.image(pr4_2)

# note that the following distributions map to the same abstract state
pr8_1 @ phi4
pr8_1a @ phi4

# as a result, belief is the same regardless of history and is therefore Markov
B4_2 = m4.B(pr8_2)
B4_2a = m4.B(pr8_2a)
assert not np.allclose(B4_2, B4_2a)

# however, T_phi does NOT depend on history, and IS Markov
T6_2 = B6_2 @ m6.base_mdp.T[0] @ phi6
T6_2a = B6_2a @ m6.base_mdp.T[0] @ phi6
assert np.allclose(T6_2, T6_2a)
