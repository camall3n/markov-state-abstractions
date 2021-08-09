import imageio
import matplotlib.pyplot as plt
import numpy as np
import seeding
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visgrid.gridworld import GridWorld, MazeWorld, SpiralWorld
from visgrid.utils import get_parser
from visgrid.sensors import *
from gridworld.models.phinet import PhiNet
from gridworld.agents.dqnagent import DQNAgent

parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
# yapf: disable
parser.add_argument('-b','--batch_size', type=int, default=32,
                    help='Number of experiences to sample per batch')
parser.add_argument('-lr','--learning_rate', type=float, default=0.0001,
                    help='Learning rate for Adam optimizer')
parser.add_argument('-s','--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--phi_path', type=str,
                    help='Load an existing abstraction network by tag')

parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
parser.add_argument('-v','--video', action='store_true',
                    help='Show video of agent training')
parser.add_argument('--rearrange_xy', action='store_true',
                    help='Rearrange discrete x-y positions to break smoothness')
parser.add_argument('--maze', action='store_true',
                    help='Add walls to the gridworld to turn it into a maze')
parser.add_argument('--spiral', action='store_true',
                    help='Add walls to the gridworld to turn it into a spiral')

if 'ipykernel' in sys.argv[0]:
    arglist = ['--maze', '--phi_path', 'test-maze-relu-no-inv']
    args = parser.parse_args(arglist)
else:
    args = parser.parse_args()

args.rows = 6
args.cols = 6
args.rearrange_xy = False
args.no_sigma = False
args.one_hot = False
args.train_phi = False
args.latent_dims = 2

if args.train_phi and args.no_phi:
    assert False, '--no_phi and --train_phi are mutually exclusive'

if args.one_hot and args.no_sigma:
    assert False, '--one_hot and --no_sigma are mutually exclusive'

if args.maze:
    env = MazeWorld.load_maze(rows=args.rows, cols=args.cols, seed=args.seed)
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
    modelfile = 'results/models/{}/phi-{}_latest.pytorch'.format(args.phi_path, args.seed)
    phinet.load(modelfile)
    phinet.eval()

seeding.seed(args.seed, np, torch)

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
    modelfile = 'results/models/{}/qnet-{}_latest.pytorch'.format(args.phi_path, args.seed)
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
n_sensor_samples = 20
zz = [phinet(torch.tensor(sensor.observe(s), dtype=torch.float32)) for _ in range(n_sensor_samples)]
q_values = torch.stack([qnet(z).max(dim=-1)[0] for z in zz]).mean(dim=0)


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

    with torch.no_grad():
        zz = [phinet(torch.tensor(sensor.observe(s), dtype=torch.float32)) for _ in range(n_sensor_samples)]
        q_values = torch.stack([qnet(z).max(dim=-1)[0] for z in zz]).mean(dim=0)

    ax[2].clear()
    ax[2].set_title('Oracle-trained NN')
    plot_value_function(q_values.detach().numpy().reshape(6,6), ax[2], v.min(), v.max())
    env.plot(ax[2])
    fig.canvas.draw()
    fig.canvas.flush_events()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

    frames.append(frame)
imageio.mimwrite('test_maze_no_inv_qnet-{}.mp4'.format(args.seed), frames, fps=30)
