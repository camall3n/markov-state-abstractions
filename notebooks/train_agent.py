import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
from tqdm import tqdm

from notebooks.phinet import PhiNet
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld
from gridworlds.utils import reset_seeds, get_parser
from gridworlds.sensors import *


parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
parser.add_argument('-n','--n_trials', type=int, default=1,
                    help='Number of trials')
parser.add_argument('-e','--n_episodes', type=int, default=10,
                    help='Number of episodes per trial')
parser.add_argument('-m','--max_steps', type=int, default=100,
                    help='Maximum number of steps per episode')
parser.add_argument('-r','--rows', type=int, default=7,
                    help='Number of gridworld rows')
parser.add_argument('-c','--cols', type=int, default=4,
                    help='Number of gridworld columns')
parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('--no_graphics', action='store_true',
                    help='Turn off graphics (e.g. for running on cluster)')
parser.set_defaults(video=False)
parser.set_defaults(no_graphics=False)
parser.set_defaults(save=False)
args = parser.parse_args()

log_dir = 'logs/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)
log = open(log_dir+'/scores-{}.txt'.format(args.seed), 'w')

reset_seeds(args.seed)

#%% ------------------ Define MDP ------------------
env = GridWorld(rows=args.rows, cols=args.cols)
sensor = SensorChain([
    OffsetSensor(offset=(0.5,0.5)),
    NoisySensor(sigma=0.05),
    ImageSensor(range=((0,env._rows), (0,env._cols)), pixel_density=3),
    # ResampleSensor(scale=2.0),
    BlurSensor(sigma=0.6, truncate=1.),
])
x0 = sensor.observe(env.get_state())

#%% ------------------ Load abstraction ------------------
modelfile = 'models/{}/phi-{}.pytorch'.format(args.tag, args.seed)
phinet = PhiNet(input_shape=x0.shape, n_latent_dims=2, n_hidden_layers=1, n_units_per_layer=32, lr=args.learning_rate)
phinet.load(modelfile)

#%% ------------------ Load agent ------------------
class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def act(self, x):
        return np.random.randint(self.n_actions)
    def train(self, x, a, r, xp, done):
        pass
agent = RandomAgent(n_actions=4)

#%% ------------------ Train agent ------------------
for trial in tqdm(range(args.n_trials), desc='trials'):
    env.reset_goal()
    total_reward = 0
    total_steps = 0
    for episode in tqdm(range(args.n_episodes), desc='episodes'):
        env.reset_agent()
        ep_rewards = []
        for step in range(args.max_steps):
            s = env.get_state()
            x = sensor.observe(s)

            a = agent.act(x)
            sp, r, done = env.step(a)
            xp = sensor.observe(sp)
            ep_rewards.append(r)
            total_reward += r

            agent.train(x, a, r, xp, done)
            if done:
                break
        total_steps += step
        score_info = {
            'trial': trial,
            'episode': episode,
            'reward': sum(ep_rewards),
            'total_reward': total_reward,
            'total_steps': total_steps,
            'steps': step
        }
        json_str = json.dumps(score_info)
        log.write(json_str+'\n')
        log.flush()
print('\n\n')
