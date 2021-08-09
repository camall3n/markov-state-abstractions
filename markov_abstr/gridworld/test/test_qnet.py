import imageio
import matplotlib.pyplot as plt
import numpy as np
import seeding
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visgrid.gridworld import GridWorld
from visgrid.sensors import *
from gridworld.models.qnet import QNet

seeding.seed(0, np, torch)

qnet = QNet(n_features=2, n_actions=4, n_hidden_layers=1, n_units_per_layer=32)
env = GridWorld(rows=6, cols=6)
gamma = 0.9
r_step = -1
r_goal = 0
env.reset_goal()
env.reset_goal()

#%%
r = np.arange(env._rows)
c = np.arange(env._cols)
s = np.asarray([[(r, c) for r in np.arange(env._rows)] for c in range(env._cols)]).reshape(-1, 2)

s.reshape(6, 6, 2)
v = -np.ones((6, 6)) * 1 / (1 - gamma)
v[3, 3] = r_goal
v_prev = v.copy()
m, n = v.shape
for i in range(1000):
    for r in range(m):
        for c in range(n):
            if c > 0:
                v[r, c] = max(v[r, c], r_step + gamma * v[r, c - 1])
            if c < m - 1:
                v[r, c] = max(v[r, c], r_step + gamma * v[r, c + 1])
            if r > 0:
                v[r, c] = max(v[r, c], r_step + gamma * v[r - 1, c])
            if r < n - 1:
                v[r, c] = max(v[r, c], r_step + gamma * v[r + 1, c])
    if np.all(np.isclose(v_prev, v)):
        break
    else:
        v_prev = v.copy()

def plot_value_function(v, ax):
    s = np.asarray([[np.asarray([x, y]) for x in range(env._cols)] for y in range(env._rows)])
    xy = OffsetSensor(offset=(0.5, 0.5)).observe(s).reshape(env._cols, env._rows, -1)
    ax.contourf(np.arange(0.5, env._cols + 0.5),
                np.arange(0.5, env._rows + 0.5),
                v,
                vmin=-10,
                vmax=0)

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
fig.show()

plot_value_function(v, ax[0])
env.plot(ax[0])
ax[0].set_title('VI solution')

optimizer = torch.optim.Adam(qnet.parameters(), lr=0.01)

frames = []
for step in tqdm(range(400)):
    qnet.train()
    optimizer.zero_grad()
    q_values = qnet(torch.tensor(s, dtype=torch.float32)).max(dim=-1)[0]
    loss = F.smooth_l1_loss(input=q_values, target=torch.from_numpy(v.reshape(-1)).float())
    loss.backward()
    optimizer.step()

    ax[1].clear()
    ax[1].set_title('Oracle-trained NN')
    plot_value_function(q_values.detach().numpy().reshape(6, 6), ax[1])
    env.plot(ax[1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    frame = np.frombuffer(fig.canvas.tostring_rgb(),
                          dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    frames.append(frame)
imageio.mimwrite('test_qnet.mp4', frames, fps=30)
