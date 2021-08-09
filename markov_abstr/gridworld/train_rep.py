import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import seeding
import sys
import torch
from tqdm import tqdm

from models.featurenet import FeatureNet
from models.autoencoder import AutoEncoder
from models.pixelpredictor import PixelPredictor
from repvis import RepVisualization, CleanVisualization
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.utils import get_parser, MI
from visgrid.sensors import *
from visgrid.gridworld.distance_oracle import DistanceOracle

parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
# yapf: disable
parser.add_argument('--type', type=str, default='markov', choices=['markov', 'autoencoder', 'pixel-predictor'],
                    help='Which type of representation learning method')
parser.add_argument('-n','--n_updates', type=int, default=3000,
                    help='Number of training updates')
parser.add_argument('-r','--rows', type=int, default=6,
                    help='Number of gridworld rows')
parser.add_argument('-c','--cols', type=int, default=6,
                    help='Number of gridworld columns')
parser.add_argument('-w', '--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop'],
                    help='The wall configuration mode of gridworld')
parser.add_argument('-l','--latent_dims', type=int, default=2,
                    help='Number of latent dimensions to use for representation')
parser.add_argument('--L_inv', type=float, default=1.0,
                    help='Coefficient for inverse-model-matching loss')
parser.add_argument('--L_coinv', type=float, default=0.0,
                    help='Coefficient for *contrastive* inverse-model-matching loss')
# parser.add_argument('--L_fwd', type=float, default=0.0,
#                     help='Coefficient for forward dynamics loss')
parser.add_argument('--L_rat', type=float, default=1.0,
                    help='Coefficient for ratio-matching loss')
# parser.add_argument('--L_fac', type=float, default=0.0,
#                     help='Coefficient for factorization loss')
parser.add_argument('--L_dis', type=float, default=0.0,
                    help='Coefficient for planning-distance loss')
parser.add_argument('--L_ora', type=float, default=0.0,
                    help='Coefficient for oracle distance loss')
parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='Mini batch size for training updates')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('-v','--video', action='store_true',
                    help="Save training video")
parser.add_argument('--no_graphics', action='store_true',
                    help='Turn off graphics (e.g. for running on cluster)')
parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
parser.add_argument('--cleanvis', action='store_true',
                    help='Switch to representation-only visualization')
parser.add_argument('--no_sigma', action='store_true',
                    help='Turn off sensors and just use true state; i.e. x=s')
parser.add_argument('--rearrange_xy', action='store_true',
                    help='Rearrange discrete x-y positions to break smoothness')

# yapf: enable
if 'ipykernel' in sys.argv[0]:
    arglist = [
        '--spiral', '--tag', 'test-spiral', '-r', '6', '-c', '6', '--L_ora', '1.0', '--video'
    ]
    args = parser.parse_args(arglist)
else:
    args = parser.parse_args()

if args.no_graphics:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_dir = 'results/logs/' + str(args.tag)
vid_dir = 'results/videos/' + str(args.tag)
maze_dir = 'results/mazes/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

if args.video:
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(maze_dir, exist_ok=True)
    video_filename = vid_dir + '/video-{}.mp4'.format(args.seed)
    image_filename = vid_dir + '/final-{}.png'.format(args.seed)
    maze_file = maze_dir + '/maze-{}.png'.format(args.seed)

log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

seeding.seed(args.seed)

#% ------------------ Define MDP ------------------
if args.walls == 'maze':
    env = MazeWorld.load_maze(rows=args.rows, cols=args.cols, seed=args.seed)
elif args.walls == 'spiral':
    env = SpiralWorld(rows=args.rows, cols=args.cols)
elif args.walls == 'loop':
    env = LoopWorld(rows=args.rows, cols=args.cols)
else:
    env = GridWorld(rows=args.rows, cols=args.cols)
# env = RingWorld(2,4)
# env = TestWorld()
# env.add_random_walls(10)

# cmap = 'Set3'
cmap = None

#% ------------------ Generate experiences ------------------
n_samples = 20000
states = [env.get_state()]
actions = []
for t in range(n_samples):
    a = np.random.choice(env.actions)
    s, _, _ = env.step(a)
    states.append(s)
    actions.append(a)
states = np.stack(states)
s0 = np.asarray(states[:-1, :])
c0 = s0[:, 0] * env._cols + s0[:, 1]
s1 = np.asarray(states[1:, :])
a = np.asarray(actions)

MI_max = MI(s0, s0)

ax = env.plot()
xx = s0[:, 1] + 0.5
yy = s0[:, 0] + 0.5
ax.scatter(xx, yy, c=c0)
if args.video:
    plt.savefig(maze_file)

# Confirm that we're covering the state space relatively evenly
# np.histogram2d(states[:,0], states[:,1], bins=6)

#% ------------------ Define sensor ------------------
sensor_list = []
if args.rearrange_xy:
    sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))
if not args.no_sigma:
    sensor_list += [
        OffsetSensor(offset=(0.5, 0.5)),
        NoisySensor(sigma=0.05),
        ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
        # ResampleSensor(scale=2.0),
        BlurSensor(sigma=0.6, truncate=1.),
        NoisySensor(sigma=0.01)
    ]
sensor = SensorChain(sensor_list)

x0 = sensor.observe(s0)
x1 = sensor.observe(s1)

#% ------------------ Setup experiment ------------------
n_updates_per_frame = 100
n_frames = args.n_updates // n_updates_per_frame

batch_size = args.batch_size

coefs = {
    'L_inv': args.L_inv,
    'L_coinv': args.L_coinv,
    # 'L_fwd': args.L_fwd,
    'L_rat': args.L_rat,
    # 'L_fac': args.L_fac,
    'L_dis': args.L_dis,
    'L_ora': args.L_ora,
}

if args.type == 'markov':
    fnet = FeatureNet(n_actions=4,
                      input_shape=x0.shape[1:],
                      n_latent_dims=args.latent_dims,
                      n_hidden_layers=1,
                      n_units_per_layer=32,
                      lr=args.learning_rate,
                      coefs=coefs)
elif args.type == 'autoencoder':
    fnet = AutoEncoder(n_actions=4,
                       input_shape=x0.shape[1:],
                       n_latent_dims=args.latent_dims,
                       n_hidden_layers=1,
                       n_units_per_layer=32,
                       lr=args.learning_rate,
                       coefs=coefs)
elif args.type == 'pixel-predictor':
    fnet = PixelPredictor(n_actions=4,
                          input_shape=x0.shape[1:],
                          n_latent_dims=args.latent_dims,
                          n_hidden_layers=1,
                          n_units_per_layer=32,
                          lr=args.learning_rate,
                          coefs=coefs)

fnet.print_summary()

n_test_samples = 2000
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]
test_x0 = torch.as_tensor(x0[-n_test_samples:, :]).float()
test_x1 = torch.as_tensor(x1[-n_test_samples:, :]).float()
test_a = torch.as_tensor(a[-n_test_samples:]).long()
test_i = torch.arange(n_test_samples).long()
test_c = c0[-n_test_samples:]

oracle = DistanceOracle(env)

env.reset_agent()
state = env.get_state()
obs = sensor.observe(state)

if args.video:
    if not args.cleanvis:
        repvis = RepVisualization(env,
                                  obs,
                                  batch_size=n_test_samples,
                                  n_dims=2,
                                  colors=test_c,
                                  cmap=cmap)
    else:
        repvis = CleanVisualization(env,
                                    obs,
                                    batch_size=n_test_samples,
                                    n_dims=2,
                                    colors=test_c,
                                    cmap=cmap)

def get_batch(x0, x1, a, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx]).float()
    tx1 = torch.as_tensor(x1[idx]).float()
    ta = torch.as_tensor(a[idx]).long()
    ti = torch.as_tensor(idx).long()
    return tx0, tx1, ta, idx

get_next_batch = (
    lambda: get_batch(x0[:n_samples // 2, :], x1[:n_samples // 2, :], a[:n_samples // 2]))

def test_rep(fnet, step):
    with torch.no_grad():
        fnet.eval()
        if args.type == 'markov':
            z0 = fnet.phi(test_x0)
            z1 = fnet.phi(test_x1)
            # z1_hat = fnet.fwd_model(z0, test_a)
            # a_hat = fnet.inv_model(z0, z1)

            # yapf: disable
            loss_info = {
                'step': step,
                'L_inv': fnet.inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_coinv': fnet.contrastive_inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_fwd': 'NaN',  #fnet.compute_fwd_loss(z0, z1, z1_hat).numpy().tolist(),
                'L_rat': fnet.ratio_loss(z0, z1).numpy().tolist(),
                'L_dis': fnet.distance_loss(z0, z1).numpy().tolist(),
                'L_fac': 'NaN',  #fnet.compute_factored_loss(z0, z1).numpy().tolist(),
                # 'L_ent': 'NaN',#fnet.compute_entropy_loss(z0, z1, test_a).numpy().tolist(),
                'L': fnet.compute_loss(z0, z1, test_a, torch.zeros((2 * len(z0)))).numpy().tolist(),
                'MI': MI(test_s0, z0.numpy()) / MI_max
            }
            # yapf: enable
        elif args.type == 'autoencoder':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'step': step,
                'L': fnet.compute_loss(test_x0).numpy().tolist(),
            }

        elif args.type == 'pixel-predictor':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'step': step,
                'L': fnet.compute_loss(test_x0, test_a, test_x1).numpy().tolist(),
            }

    json_str = json.dumps(loss_info)
    log.write(json_str + '\n')
    log.flush()

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    results = [z0, z1, z1, test_a, test_a]
    return [r.numpy() for r in results] + [text]

#% ------------------ Run Experiment ------------------
data = []
for frame_idx in tqdm(range(n_frames + 1)):
    for _ in range(n_updates_per_frame):
        tx0, tx1, ta, idx = get_next_batch()
        tdist = torch.cat([
            torch.as_tensor(oracle.pairwise_distances(idx, s0, s1)).squeeze().float(),
            torch.as_tensor(oracle.pairwise_distances(idx, s0, np.flip(s1))).squeeze().float()
        ], dim=0) # yapf: disable
        # h = np.histogram(tdist, bins=36)[0]

        fnet.train_batch(tx0, tx1, ta, tdist)

    test_results = test_rep(fnet, frame_idx * n_updates_per_frame)
    if args.video:
        frame = repvis.update_plots(*test_results)
        data.append(frame)

if args.video:
    imageio.mimwrite(video_filename, data, fps=15)
    imageio.imwrite(image_filename, data[-1])

if args.save:
    fnet.phi.save('phi-{}'.format(args.seed), 'results/models/{}'.format(args.tag))

log.close()
