import time

from tqdm import tqdm
# ---------------------------------------------------------------
# This hack prevents a matplotlib/dm_control crash on macOS.
# We open/close a plot before importing dm_control.suite, which
# in this case happens when we import gym and make a dm2gym env.
# import matplotlib.pyplot as plt
# plt.plot()
# plt.close()
import gym
# ---------------------------------------------------------------

import imageio

env = gym.make('dm2gym:CartpoleSwingup-v0')

# plt.ion()
# plt.show()

env.reset()
trials = 1000
s = time.time()
imgs = []
for i in tqdm(range(trials)):
    _, _, done, _ = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array', use_opencv_renderer=True)
    imgs.append(img)
    # plt.imshow(img)
    # plt.draw()
    # plt.pause(1 / 60)
    if done:
        env.reset()
e = time.time()

print("Speed: {}".format(trials / (e - s)), 'iterations per second')

# plt.show()
env.close()

imageio.mimwrite('test_dm2gym.mp4', imgs, fps=60)
