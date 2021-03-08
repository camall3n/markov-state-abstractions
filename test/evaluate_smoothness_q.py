import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gridworlds.domain.gridworld.gridworld import GridWorld, MazeWorld, SpiralWorld
from gridworlds.nn.phinet import PhiNet
from gridworlds.agents.dqnagent import DQNAgent
from gridworlds.nn.qnet import FactoredQNet
from gridworlds.utils import reset_seeds
from gridworlds.sensors import *

class Args: pass
args = Args()
args.rows = 6
args.cols = 6
args.maze = False
args.spiral = True
args.rearrange_xy = False
args.no_sigma = False
args.one_hot = False
args.phi_path = 'test-spiral-distloss-10'
args.train_phi = False
args.seed = 1
args.latent_dims = 2
args.learning_rate = 0.0001
args.batch_size = 32


if args.maze:
    env = MazeWorld(rows=args.rows, cols=args.cols)
elif args.spiral:
    env = SpiralWorld(rows=args.rows, cols=args.cols)
else:
    env = GridWorld(rows=args.rows, cols=args.cols)
gamma = 0.9

#% ------------------ Define sensor ------------------
sensor_list = []
if args.rearrange_xy:
    sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))
if not args.no_sigma:
    if args.one_hot:
        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=1),
        ]
    else:
        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            NoisySensor(sigma=0.05),
            ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
            # ResampleSensor(scale=2.0),
            BlurSensor(sigma=0.6, truncate=1.),
            NoisySensor(sigma=0.01)
        ]
sensor = SensorChain(sensor_list)

#% ------------------ Define abstraction ------------------

x0 = sensor.observe(env.get_state())
phinet = PhiNet(input_shape=x0.shape,
                n_latent_dims=args.latent_dims,
                n_hidden_layers=1,
                n_units_per_layer=32)
if args.phi_path:
    modelfile = 'models/{}/phi-{}_latest.pytorch'.format(args.phi_path, args.seed)
    phinet.load(modelfile)
    phinet.eval()

reset_seeds(args.seed)

n_actions = 4
gamma = 0.9
agent = DQNAgent(n_features=args.latent_dims,
                 n_actions=n_actions,
                 phi=phinet,
                 lr=args.learning_rate,
                 batch_size=args.batch_size,
                 train_phi=args.train_phi,
                 gamma=gamma,
                 factored=False)
if args.phi_path:
    modelfile = 'models/{}/qnet-{}_latest.pytorch'.format(args.phi_path, args.seed)
    agent.q.load(modelfile)
qnet = agent.q


r_step = -1
r_goal = 0
env.reset_goal()

env.goal.position

#%%
r = np.arange(env._rows)
c = np.arange(env._cols)
s = np.asarray([[(r,c) for r in np.arange(env._rows)] for c in range(env._cols)]).reshape(-1,2)

s.reshape(6,6,2)
v = -np.ones((6,6))*1/(1-gamma)
v[tuple(env.goal.position)] = r_goal
v_prev = v.copy()
m,n = v.shape
for i in range(1000):
    for r in range(m):
        for c in range(n):
            if c > 0 and not env.has_wall((r, c), (0, -1)):
                v[r,c] = max(v[r,c], r_step + gamma * v[r,c-1])
            if c < m-1 and not env.has_wall((r, c), (0, 1)):
                v[r,c] = max(v[r,c], r_step + gamma * v[r,c+1])
            if r > 0 and not env.has_wall((r, c), (-1, 0)):
                v[r,c] = max(v[r,c], r_step + gamma * v[r-1,c])
            if r < n-1 and not env.has_wall((r, c), (1, 0)):
                v[r,c] = max(v[r,c], r_step + gamma * v[r+1,c])
    if np.all(np.isclose(v_prev,v)):
        break
    else:
        v_prev = v.copy()

def plot_value_function(v, ax, vmin, vmax):
    x = np.arange(0, env._cols)
    xx = np.concatenate([x, x+.98])
    xx.sort()
    y = np.arange(0, env._rows)
    yy = np.concatenate([y, y+.98])
    yy.sort()
    xx, yy = np.meshgrid(xx, yy)
    vv = np.round(v)
    vv = np.repeat(np.repeat(vv, 2, 1), 2, 0)
    ax.contourf(xx, yy, vv, vmin=vmin, vmax=vmax, cmap='plasma')

fig, ax = plt.subplots(1,3, figsize=(6,3))
plt.ion()
fig.show()

plot_value_function(v, ax[0], v.min(), v.max())
env.plot(ax[0])
ax[0].set_title('VI solution')
plt.show()


qnet.eval()
z = phinet(torch.tensor(sensor.observe(s), dtype=torch.float32))
q_values = qnet(z).max(dim=-1)[0]

ax[1].clear()
ax[1].set_title('Learned Q')
plot_value_function(q_values.detach().numpy().reshape(6,6), ax[1], v.min(), v.max())
env.plot(ax[1])
fig.canvas.draw()
fig.canvas.flush_events()
frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

# plt.savefig('evaluate_smoothness_q.png')
# plt.show()


#%%
optimizer = torch.optim.Adam(qnet.parameters(), lr=0.01)

frames=[]
for step in tqdm(range(400)):
    qnet.train()
    optimizer.zero_grad()
    with torch.no_grad():
        z = phinet(torch.tensor(sensor.observe(s), dtype=torch.float32))
    q_values = qnet(z).max(dim=-1)[0]
    loss = F.smooth_l1_loss(input=q_values, target=torch.from_numpy(v.reshape(-1)).float())
    loss.backward()
    optimizer.step()

    ax[2].clear()
    ax[2].set_title('Oracle-trained NN')
    plot_value_function(q_values.detach().numpy().reshape(6,6), ax[2], v.min(), v.max())
    env.plot(ax[2])
    fig.canvas.draw()
    fig.canvas.flush_events()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

    frames.append(frame)
imageio.mimwrite('test_qnet.mp4', frames, fps=30)
