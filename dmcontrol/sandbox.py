from tqdm import tqdm
# ---------------------------------------------------------------
# This hack prevents a matplotlib/dm_control crash on macOS.
# We open/close a plot before importing dm_control.suite, which
# in this case happens when we import gym and make a dm2gym env.
import matplotlib.pyplot as plt
plt.plot()
plt.close()
import gym
# ---------------------------------------------------------------

import imageio
import numpy as np

from dmcontrol import gym_wrappers as wrap
from dmcontrol import rad

#%%

env = gym.make('dm2gym:WalkerWalk-v0', environment_kwargs={'flat_observation': True})

def wrap_env(env, feature_type='markov', size=(84, 84), action_repeat=1, frame_stack=3):
    env = wrap.FixedDurationHack(env)
    env = wrap.ObservationDictToInfo(env, "observations")

    if feature_type == 'expert':
        env = wrap.MaxAndSkipEnv(env, skip=action_repeat, max_pool=False)
    else:
        env = wrap.RenderOpenCV(env)
        env = wrap.Grayscale(env)
        env = wrap.ResizeObservation(env, size)
        env = wrap.MaxAndSkipEnv(env, skip=action_repeat, max_pool=False)
        env = wrap.FrameStack(env, frame_stack, lazy=False)
    return env

env = wrap_env(env)
weights = np.expand_dims(np.expand_dims(np.asarray([1, 2, 3]), -1), -1) / 6

state = env.reset()
state = env.step(env.action_space.sample())[0]
plt.imshow((state * weights).sum(axis=0))
plt.imshow(state.std(axis=0))

n_steps = 100
imgs = []
for i in tqdm(range(n_steps)):
    state, _, done, _ = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array', use_opencv_renderer=True)
    imgs.append(img)
    if done:
        break

#%% CURL testing
env = gym.make('dm2gym:WalkerWalk-v0', environment_kwargs={'flat_observation': True})
env = wrap_env(env, size=(100, 100))
weights = np.expand_dims(np.expand_dims(np.asarray([1, 2, 3]), -1), -1) / 6

state = env.reset()
state = env.step(env.action_space.sample())[0]

batch_size = 10
states = np.asarray([env.step(env.action_space.sample())[0] for _ in range(batch_size)])
state = states[0]

plt.imshow((state * weights).sum(axis=0))

crop = rad.center_crop_one_image(states[0])
plt.imshow((crop * weights).sum(axis=0))


crop = rad.random_crop_image_batch(states)[0]
plt.imshow((crop * weights).sum(axis=0))
