import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import torch
from tqdm import tqdm

from gridworlds.nn.featurenet import FeatureNet
from gridworlds.nn.autoencoder import AutoEncoder
from notebooks.repvis import RepVisualization, CleanVisualization
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld
from gridworlds.utils import reset_seeds, get_parser, MI
from gridworlds.sensors import *

parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
# yapf: disable
parser.add_argument('--type', type=str, default='markov', choices=['markov', 'autoencoder'],
                    help='Which type of representation learning method')
parser.add_argument('-n','--n_updates', type=int, default=10000,
                    help='Number of training updates')
parser.add_argument('-r','--rows', type=int, default=7,
                    help='Number of gridworld rows')
parser.add_argument('-c','--cols', type=int, default=4,
                    help='Number of gridworld columns')
parser.add_argument('-l','--latent_dims', type=int, default=2,
                    help='Number of latent dimensions to use for representation')
parser.add_argument('--L_inv', type=float, default=1.0,
                    help='Coefficient for inverse-model-matching loss')
# parser.add_argument('--L_fwd', type=float, default=0.0,
#                     help='Coefficient for forward dynamics loss')
parser.add_argument('--L_rat', type=float, default=1.0,
                    help='Coefficient for ratio-matching loss')
# parser.add_argument('--L_fac', type=float, default=0.0,
#                     help='Coefficient for factorization loss')
parser.add_argument('--L_dis', type=float, default=0.0,
                    help='Coefficient for planning-distance loss')
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
# yapf: enable
args = parser.parse_args()

if args.no_graphics:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')

log_dir = 'logs/' + str(args.tag)
vid_dir = 'videos/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

if args.video:
    os.makedirs(vid_dir, exist_ok=True)
    filename = vid_dir + '/video-{}.mp4'.format(args.seed)

log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

reset_seeds(args.seed)

#%% ------------------ Define MDP ------------------
env = GridWorld(rows=args.rows, cols=args.cols)
# env = RingWorld(2,4)
# env = TestWorld()
# env.add_random_walls(10)
# env.plot()

# cmap = 'Set3'
cmap = None

#%% ------------------ Generate experiences ------------------
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

#%% ------------------ Define sensor ------------------
sensor = SensorChain([
    OffsetSensor(offset=(0.5, 0.5)),
    NoisySensor(sigma=0.05),
    ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
    # ResampleSensor(scale=2.0),
    BlurSensor(sigma=0.6, truncate=1.),
    NoisySensor(sigma=0.01)
])

x0 = sensor.observe(s0)
x1 = sensor.observe(s1)

#%% ------------------ Setup experiment ------------------
n_updates_per_frame = 100
n_frames = args.n_updates // n_updates_per_frame

batch_size = args.batch_size

coefs = {
    'L_inv': args.L_inv,
    # 'L_fwd': args.L_fwd,
    'L_rat': args.L_rat,
    # 'L_fac': args.L_fac,
    'L_dis': args.L_dis,
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

fnet.print_summary()

n_test_samples = 2000
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]
test_x0 = torch.as_tensor(x0[-n_test_samples:, :]).float()
test_x1 = torch.as_tensor(x1[-n_test_samples:, :]).float()
test_a = torch.as_tensor(a[-n_test_samples:]).long()
test_i = torch.arange(n_test_samples).long()
test_c = c0[-n_test_samples:]

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
    return tx0, tx1, ta, ti

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

            loss_info = {
                'step': step,
                'L_inv': fnet.inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_fwd': 'NaN',  #fnet.compute_fwd_loss(z0, z1, z1_hat).numpy().tolist(),
                'L_rat': fnet.ratio_loss(z0, z1).numpy().tolist(),
                'L_dis': fnet.distance_loss(z0, z1, test_i).numpy().tolist(),
                'L_fac': 'NaN',  #fnet.compute_factored_loss(z0, z1).numpy().tolist(),
                # 'L_ent': 'NaN',#fnet.compute_entropy_loss(z0, z1, test_a).numpy().tolist(),
                'L': fnet.compute_loss(z0, z1, test_a, test_i, 'all').numpy().tolist(),
                'MI': MI(test_s0, z0.numpy()) / MI_max
            }
        elif args.type == 'autoencoder':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'step': step,
                'L': fnet.compute_loss(test_x0).numpy().tolist(),
            }

    json_str = json.dumps(loss_info)
    log.write(json_str + '\n')
    log.flush()

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    results = [z0, z1, z1, test_a, test_a]
    return [r.numpy() for r in results] + [text]

#%% ------------------ Run Experiment ------------------
data = []
for frame_idx in tqdm(range(n_frames + 1)):
    for _ in range(n_updates_per_frame):
        tx0, tx1, ta, ti = get_next_batch()
        fnet.train_batch(tx0, tx1, ta, ti, model='all')

    test_results = test_rep(fnet, frame_idx * n_updates_per_frame)
    if args.video:
        frame = repvis.update_plots(*test_results)
        data.append(frame)

if args.video:
    imageio.mimwrite(filename, data, fps=15)

if args.save:
    fnet.phi.save(args.tag, 'phi-{}'.format(args.seed))

log.close()
