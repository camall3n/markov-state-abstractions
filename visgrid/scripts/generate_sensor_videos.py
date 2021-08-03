import imageio
import numpy as np
import matplotlib.pyplot as plt
import seeding
from tqdm import tqdm

from visgrid.gridworld import GridWorld
from visgrid.sensors import *

env = GridWorld(rows=6,cols=6)
env.reset_agent()

sensor = SensorChain([
    OffsetSensor(offset=(0.5,0.5)),
    NoisySensor(sigma=0.05),
    ImageSensor(range=((0,env._rows), (0,env._cols)), pixel_density=3),
    # ResampleSensor(scale=2.0),
    BlurSensor(sigma=0.6, truncate=1.),
    NoisySensor(sigma=0.01)
])

#%%
%matplotlib agg
seeding.seed(0, np)
env.reset_agent()
fig, ax = plt.subplots(figsize=(16,16))
jointfig, jointax = plt.subplots(1,2,figsize=(32,16))
ax.axis('off')
[a.axis('off') for a in jointax]
s = []
s_frames = []
x_frames = []
xs_frames = []
fwd_actions = np.array([3, 0, 2, 0, 3, 2, 2, 0, 2, 0, 3, 1, 0, 2, 2, 0, 2, 2, 0])
inv_actions = np.array([{0: 1, 1: 0, 2: 3, 3:2}[x] for x in fwd_actions])
actions = np.concatenate([fwd_actions, inv_actions])
for a in tqdm(actions):
    env.step(a)
    s.append(env.get_state())
    ax.clear()
    jointax[1].clear()
    env.plot(ax=ax)
    env.plot(ax=jointax[1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
    s_frames.append(frame)
    for t in range(5):
        ax.clear()
        jointax[0].clear()
        x = sensor.observe(env.get_state())
        ax.imshow(x)
        jointax[0].imshow(x)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        [a.set_xticks([]) for a in jointax]
        [a.set_yticks([]) for a in jointax]
        fig.canvas.draw()
        fig.canvas.flush_events()
        jointfig.canvas.draw()
        jointfig.canvas.flush_events()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        jointframe = np.frombuffer(jointfig.canvas.tostring_rgb(), dtype=np.uint8).reshape(jointfig.canvas.get_width_height()[::-1] + (3,))
        x_frames.append(frame)
        xs_frames.append(jointframe)

imageio.mimwrite('x_frames.mp4', x_frames, fps=25)
# imageio.mimwrite('s_frames.mp4', s_frames, fps=5)
# imageio.mimwrite('xs_frames.mp4', xs_frames, fps=25)

imageio.imwrite('x_frame0.png', x_frames[0])
imageio.imwrite('s_frame0.png', s_frames[0])

#%%
# env.plot()
for i in range(10):
    s = env.get_state()
    obs = sensor.observe(s)

    plt.figure()
    plt.imshow(obs)
    plt.xticks([])
    plt.yticks([])
    plt.show()
