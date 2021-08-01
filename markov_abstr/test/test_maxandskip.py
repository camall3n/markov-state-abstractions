import copy

import gym

from dmcontrol.gym_wrappers import MaxAndSkipEnv

def set_random_seed(seed, env):
    import numpy
    numpy.random.seed(seed)
    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    env.seed(seed)
    env.unwrapped.np_random.seed(seed)
    env.action_space.seed(seed)

env1 = gym.make('Pendulum-v0')
env2 = gym.make('Pendulum-v0')
env2 = MaxAndSkipEnv(env2, skip=1, max_pool=False)

set_random_seed(0, env1)
set_random_seed(0, env1)
set_random_seed(0, env2)
env1.reset()
env2.reset()
assert (env1.reset() == env2.reset()).all()

for t in range(2000):
    action1 = env1.action_space.sample()
    action2 = env2.action_space.sample()
    assert (action1 == action2).all()

    obs1, reward1, done1, info1 = env1.step(action1)
    obs2, reward2, done2, info2 = env2.step(action2)
    assert (obs1 == obs2).all()
    assert (reward1 == reward2).all()

    assert (done1 == done2)
    if done1:
        env1.reset()
        env2.reset()

print('All tests passed.')
