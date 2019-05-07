import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from gridworlds.nn.nullabstraction import NullAbstraction
from gridworlds.nn.factornet import FactorNet
from gridworlds.domain.gridworld.gridworld import GridWorld
from gridworlds.utils import reset_seeds, get_parser, MI
from gridworlds.sensors import *

#%% ------------------ Define MDP ------------------
reset_seeds(0)

env = GridWorld(rows=6,cols=6)
env.reset_agent()

sensor = SensorChain([
    # OffsetSensor(offset=(0.5,0.5)),
    # NoisySensor(sigma=0.1),
    # ImageSensor(range=((0,env._rows), (0,env._cols)), pixel_density=1),
    # ResampleSensor(scale=(2,1)),
    # BlurSensor(sigma=0.6, truncate=1.),
    # NoisySensor(sigma=0.05),
    TorchSensor(),
])
phi = NullAbstraction(-1, 2)

#%% ------------------ Generate experiences ------------------
n_samples = 1000
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
s1 = np.asarray(states[1:,:])
c0 = s0[:,0]*env._cols+s0[:,1]
a = np.asarray(actions)

MI_max = MI(s0,s0)

z0 = phi(sensor.observe(s0))
z1 = phi(sensor.observe(s1))

entangler = FactorNet(lr=0.03, coefs={'L_fac': -0.1})
disentangler = FactorNet(lr=0.03, coefs={'L_fac': 0.1})

#%% ------------------ Train entangler ------------------
for update in tqdm(range(1000)):
    entangler.train_batch(z0, z1)
e0 = entangler(z0)
e1 = entangler(z1)

#%% ------------------ Train disentangler ------------------
for update in tqdm(range(1000)):
    disentangler.train_batch(e0, e1)
d0 = disentangler(e0)
d1 = disentangler(e1)

#%%
def plot2d(rep, title, save=''):
    rep = rep.detach().numpy()
    plt.scatter(rep[:,0], rep[:,1], c=c0)
    plt.title(title)
    if save != '':
        plt.savefig(save)
    plt.show()

plot2d(z0, title='True state (MI=1.0)', save='results/factornet/img1-true_state.png')
e_title = 'Entangled (MI={})'.format(MI(s0, e0.detach().numpy())/MI_max)
plot2d(e0, title=e_title, save='results/factornet/img2-entangled.png')
d_title = 'Disentangled (MI={})'.format(MI(s0, d0.detach().numpy())/MI_max)
plot2d(d0, title=d_title, save='results/factornet/img3-disentangled.png')
