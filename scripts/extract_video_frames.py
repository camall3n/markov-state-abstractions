import glob
import os

import imageio
import numpy as np
import matplotlib.pyplot as plt

#%%

def crop(frame):
    left_crop = 71
    right_crop = 35
    top_crop = 76
    bottom_crop = 70
    return frame[top_crop:-bottom_crop, left_crop:-right_crop]

def pad(frame):
    n = padding
    color = 255
    return np.pad(frame, ((n, n), (n, n), (0, 0)), constant_values=color)

def extract_frames(video, frames):
    return [pad(crop(video[frame])) for frame in frames]

#%%
frames = [0, 1, 2, 7, 30, 100, 300]
padding = 20

output_dir = 'results/paper/contact-sheets/'
os.makedirs(output_dir, exist_ok=True)

for seed in range(1, 7):
    rows = []
    for row, rep in enumerate(['markov', 'inv-only', 'contr-only', 'autoenc', 'pixel-pred']):
        for filename in glob.glob('results/paper/selected-videos/{}/video-{}.mp4'.format(
                rep, seed)):
            video = imageio.mimread(filename, memtest=False)
            stills = extract_frames(video, frames)
            rows.append(np.hstack(stills))
    img = np.vstack(rows)
    imageio.imwrite(output_dir + 'contact-sheet-{}.png'.format(seed), img)
