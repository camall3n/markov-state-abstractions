import time

import cv2
from PIL import Image
import scipy
import numpy as np
# import skimage

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
n_steps = 10
s = time.time()
imgs = []
for i in tqdm(range(n_steps)):
    _, _, done, _ = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array', use_opencv_renderer=True)
    imgs.append(img)
    # plt.imshow(img)
    # plt.draw()
    # plt.pause(1 / 60)
    if done:
        env.reset()
e = time.time()
print("Speed: {}".format(n_steps / (e - s)), 'iterations per second')

# plt.show()
env.close()

print()

s = time.time()
for img in imgs:
    other = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
e = time.time()
print("cv2: {}".format(n_steps / (e - s)), 'conversions per second')

s = time.time()
for img in imgs:
    other = Image.fromarray(img).convert('L')
e = time.time()
print("PIL: {}".format(n_steps / (e - s)), 'conversions per second')

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(int)

s = time.time()
for img in imgs:
    other = rgb2gray(img)
e = time.time()
print("np: {}".format(n_steps / (e - s)), 'conversions per second')

img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_pil = Image.fromarray(img).convert('L')
img_np = rgb2gray(img)

print('(img_cv2 == img_pil)', np.all(img_cv2 == img_pil))
print('(img_pil == img_np)', np.all(img_pil == img_np))
print('(img_np == img_cv2)', np.all(img_np == img_cv2))

# s = time.time()
# for img in imgs:
#     other = skimage.color.rgb2gray(img)
# e = time.time()
# print("skimage: {}".format(n_steps / (e - s)), 'conversions per second')
