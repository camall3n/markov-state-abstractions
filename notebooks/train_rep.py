import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import torch
from tqdm import tqdm

from notebooks.featurenet import FeatureNet
from notebooks.repvis import RepVisualization
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld
from gridworlds.utils import reset_seeds, get_parser, MI
from gridworlds.sensors import *


parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
parser.add_argument('-n','--n_updates', type=int, default=3000,
                    help='Number of training updates')
parser.add_argument('-r','--rows', type=int, default=7,
                    help='Number of gridworld rows')
parser.add_argument('-c','--cols', type=int, default=4,
                    help='Number of gridworld columns')
parser.add_argument('--L_inv', type=float, default=1.0,
                    help='Coefficient for inverse dynamics loss')
parser.add_argument('--L_fwd', type=float, default=0.1,
                    help='Coefficient for forward dynamics loss')
parser.add_argument('--L_cpc', type=float, default=1.0,
                    help='Coefficient for contrastive predictive coding loss')
parser.add_argument('--L_fac', type=float, default=0.1,
                    help='Coefficient for factorization loss')
parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('-v','--video', action='store_true',
                    help="Save training video")
parser.add_argument('--no_graphics', action='store_true',
                    help='Turn off graphics (e.g. for running on cluster)')
parser.set_defaults(video=False)
parser.set_defaults(no_graphics=False)
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
    filename = vid_dir+'/video-{}.mp4'.format(args.seed)

log = open(log_dir+'/train-{}.txt'.format(args.seed), 'w')
with open(log_dir+'/args-{}.txt'.format(args.seed), 'w') as arg_file:
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
    while True:
        a = np.random.choice(env.actions)
        if env.can_run(a):
            break
    s, _, _ = env.step(a)
    states.append(s)
    actions.append(a)
states = np.stack(states)
s0 = np.asarray(states[:-1,:])
c0 = s0[:,0]*env._cols+s0[:,1]
s1 = np.asarray(states[1:,:])
a = np.asarray(actions)

MI_max = MI(s0,s0)

#%% ------------------ Define sensor ------------------
sensor = SensorChain([
    OffsetSensor(offset=(0.5,0.5)),
    NoisySensor(sigma=0.05),
    ImageSensor(range=((0,env._rows), (0,env._cols)), pixel_density=3),
    # ResampleSensor(scale=2.0),
    BlurSensor(sigma=0.6, truncate=1.),
])

x0 = sensor.observe(s0)
x1 = sensor.observe(s1)

#%% ------------------ Setup experiment ------------------
n_updates_per_frame = 100
n_frames = args.n_updates // n_updates_per_frame

batch_size = 1024

coefs = {
    'L_inv': args.L_inv,
    'L_fwd': args.L_fwd,
    'L_cpc': args.L_cpc,
    'L_fac': args.L_fac,
}

fnet = FeatureNet(n_actions=4, input_shape=x0.shape[1:], n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=args.learning_rate, coefs=coefs)
fnet.print_summary()

n_test_samples = 2000
test_s0 = s0[-n_test_samples:,:]
test_s1 = s1[-n_test_samples:,:]
test_x0 = torch.as_tensor(x0[-n_test_samples:,:], dtype=torch.float32)
test_x1 = torch.as_tensor(x1[-n_test_samples:,:], dtype=torch.float32)
test_a  = torch.as_tensor(a[-n_test_samples:], dtype=torch.long)
test_c  = c0[-n_test_samples:]

env.reset_agent()
state = env.get_state()
obs = sensor.observe(state)

if args.video:
    repvis = RepVisualization(env, obs, batch_size=n_test_samples, n_dims=2, colors=test_c, cmap=cmap)

def get_batch(x0, x1, a, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx], dtype=torch.float32)
    tx1 = torch.as_tensor(x1[idx], dtype=torch.float32)
    ta = torch.as_tensor(a[idx])
    return tx0, tx1, ta

get_next_batch = lambda: get_batch(x0[:n_samples//2,:], x1[:n_samples//2,:], a[:n_samples//2])

def test_rep(fnet, step):
    with torch.no_grad():
        fnet.eval()
        z0 = fnet.phi(test_x0)
        z1 = fnet.phi(test_x1)
        z1_hat = fnet.fwd_model(z0, test_a)
        a_hat = fnet.inv_model(z0, z1)

        loss_info = {
            'step': step,
            'L_inv': fnet.compute_inv_loss(a_logits=a_hat, a=test_a).numpy().tolist(),
            'L_fwd': fnet.compute_fwd_loss(z0, z1, z1_hat).numpy().tolist(),
            'L_cpc': fnet.compute_cpc_loss(z1, z1_hat).numpy().tolist(),
            'L_fac': fnet.compute_factored_loss(z0, z1).numpy().tolist(),
            'L_ent': fnet.compute_entropy_loss(z0, z1, test_a).numpy().tolist(),
            'L': fnet.compute_loss(z0, z1, z1_hat, test_a, 'all').numpy().tolist(),
            'MI': MI(test_s0, z0.numpy())/MI_max
        }
        json_str = json.dumps(loss_info)
        log.write(json_str+'\n')
        log.flush()

        text = '\n'.join([key+' = '+str(val) for key, val in loss_info.items()])

    results = [z0, z1_hat, z1, test_a, a_hat]
    return [r.numpy() for r in results] + [text]

#%% ------------------ Run Experiment ------------------
data = []
for frame_idx in tqdm(range(n_frames+1)):
    for _ in range(n_updates_per_frame):
        tx0, tx1, ta = get_next_batch()
        fnet.train_batch(tx0, tx1, ta, model='all')

    test_results = test_rep(fnet, frame_idx*n_updates_per_frame)
    if args.video:
        frame = repvis.update_plots(*test_results)
        data.append(frame)

if args.video:
    imageio.mimwrite(filename, data, fps=15)

log.close()
